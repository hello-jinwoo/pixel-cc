import torch,os,random,time,rawpy,json

from tqdm import tqdm
from torch import optim
from torch import nn
from model import U_Net
from utils import apply_wb
from metrics import *
from torch.utils.tensorboard import SummaryWriter

class Solver():
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Misc
        self.start_time = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_interval = config.log_interval

        # RAW args
        self.camera = config.camera
        self.raw = rawpy.imread(self.camera+".dng")
        self.white_level = self.raw.white_level
        if self.camera == 'sony':
            self.white_level = self.white_level/4

        # Training config
        self.mode = config.mode
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.criterion = nn.MSELoss(reduction='mean')
        self.num_epochs_decay = config.num_epochs_decay

        # Data loader
        self.data_root = config.data_root
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.input_type = config.input_type
        self.output_type = config.output_type

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.checkpoint = config.checkpoint

        # Visualize step
        self.save_result = config.save_result
        self.vis_step = config.vis_step

        # Path
        if self.mode == 'train':
            self.model_path = os.path.join(config.model_root,self.start_time)
            self.result_path = os.path.join(config.result_root,self.start_time+'_train')
            self.log_path = os.path.join(config.log_root,self.start_time)
            self.writer = SummaryWriter(self.log_path)
        elif self.mode == 'test':
            self.model_path = os.path.join(config.model_root,self.checkpoint)
            self.result_path = os.path.join(config.result_root,self.checkpoint+'_test')
        
        if os.path.isdir(self.model_path) == False and self.mode == 'train':
            os.makedirs(self.model_path)
        if os.path.isdir(self.result_path) == False and self.save_result == 'yes':
            os.makedirs(self.result_path)

        with open(os.path.join(self.model_path,'args.txt'), 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        f.close()

        self.runninginfo = self.start_time+'_'+self.data_root+'_'+'_'.join(str(config.image_pool))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        # build network, configure optimizer
        print("[Model]\tBuilding Unet...")

        self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)

        if self.mode == 'test': # load model from checkpoint
            ckpt = os.path.join(self.model_path,'best.pt')
            print("[Model]\tLoad model from checkpoint :", ckpt)
            self.unet.load_state_dict(torch.load(ckpt))

        # multi-GPU
        if torch.cuda.device_count() > 1:
            self.unet = nn.DataParallel(self.unet)
        
        # GPU & optimizer
        self.unet.to(self.device)
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        print("[Model]\tBuild Complete.")

    def train(self):
        print("[Train]\tStart training process.")
        best_unet_score = 9876543210.
        
        for epoch in range(self.num_epochs):
            self.unet.train(True)
            AE = 0
            trainbatch_len = len(self.train_loader)

            for i, batch in enumerate(self.train_loader):
                # prepare input
                if self.input_type == "rgb":
                    inputs = batch["input_rgb"].to(self.device)
                elif self.input_type == "uvl":
                    inputs = batch["input_uvl"].to(self.device)
                # prepare GT
                if self.output_type == "illumination":
                    GTs = batch["gt_illum"].to(self.device)
                elif self.output_type == "uv":
                    GTs = batch["gt_uv"].to(self.device)
                # prepare mask
                masks = batch["mask"].to(self.device)

                # inference
                pred = self.unet(inputs)
                pred_cpu = pred.detach().cpu()
                loss = self.criterion(pred*masks, GTs*masks)

                # Backprop & optimize network
                self.unet.zero_grad()
                loss.backward()
                self.optimizer.step()

                # calculate pred_rgb & pred_illum & gt_illum
                if self.output_type == "illumination":
                    ones = torch.ones_like(pred_cpu[:,:1,:,:])
                    pred_illum = torch.cat([pred_cpu[:,:1,:,:],ones,pred_cpu[:,1:,:,:]],dim=1)
                    pred_rgb = apply_wb(batch["input_rgb"],pred_illum,pred_type='illumination')
                elif self.output_type == "uv":
                    pred_rgb = apply_wb(batch["input_rgb"],pred_cpu,pred_type='uv')
                    pred_illum = batch["input_rgb"] / (pred_rgb + 1e-8)
                ones = torch.ones_like(batch["gt_illum"][:,:1,:,:])
                gt_illum = torch.cat([batch["gt_illum"][:,:1,:,:],ones,batch["gt_illum"][:,1:,:,:]],dim=1)

                if i % self.log_interval == 0:
                    # error metrics
                    MAE_illum = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
                    MAE_rgb = get_MAE(pred_rgb,batch["gt_rgb"],tensor_type="rgb",camera=self.camera,mask=batch["mask"])
                    PSNR = get_PSNR(pred_rgb,batch["gt_rgb"],white_level=self.white_level)

                    # print training log & write on tensorboard & reset vriables
                    print(f'[Train] Epoch [{epoch+1} / {self.num_epochs}] | ' \
                            f'Batch [{i+1} / {trainbatch_len}] | ' \
                            f'loss: {loss.item():.6f} | ' \
                            f'MAE_illum: {MAE_illum:.6f} | '\
                            f'MAE_rgb: {MAE_rgb:.6f} | '\
                            f'PSNR: {PSNR:.6f}')
                    self.writer.add_scalar('train/loss', \
                                            loss.item(), \
                                            epoch * trainbatch_len + i)
                    self.writer.add_scalar('train/MAE_illum', \
                                            MAE_illum, \
                                            epoch * trainbatch_len + i)
                    self.writer.add_scalar('train/MAE_rgb', \
                                            MAE_rgb, \
                                            epoch * trainbatch_len + i)
                    self.writer.add_scalar('train/PSNR', \
                                            PSNR, \
                                            epoch * trainbatch_len + i)

            # lr decay
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                self.lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f'Decay lr to {self.lr}')
            
            # ============================================================================ #
            #                                 Validation                                   #
            # ============================================================================ #
            self.unet.eval()

            total_loss = 0
            total_len = 0
            total_mae_illum = 0
            total_mae_rgb = 0
            total_psnr = 0

            for i, batch in enumerate(self.valid_loader):
                # prepare input
                if self.input_type == "rgb":
                    inputs = batch["input_rgb"].to(self.device)
                elif self.input_type == "uvl":
                    inputs = batch["input_uvl"].to(self.device)
                # prepare GT
                if self.output_type == "illumination":
                    GTs = batch["gt_illum"].to(self.device)
                elif self.output_type == "uv":
                    GTs = batch["gt_uv"].to(self.device)
                # prepare mask
                masks = batch["mask"].to(self.device)                

                # inference
                pred = self.unet(inputs)
                pred_cpu = pred.detach().cpu()

                validbatch_len = len(inputs)
                total_len += validbatch_len

                loss = self.criterion(pred, GTs)
                total_loss += loss.item() * validbatch_len

                # calculate pred_rgb & pred_illum & gt_illum
                if self.output_type == "illumination":
                    ones = torch.ones_like(pred_cpu[:,:1,:,:])
                    pred_illum = torch.cat([pred_cpu[:,:1,:,:],ones,pred_cpu[:,1:,:,:]],dim=1)
                    pred_rgb = apply_wb(batch["input_rgb"],pred_illum,pred_type='illumination')
                elif self.output_type == "uv":
                    pred_rgb = apply_wb(batch["input_rgb"],pred_cpu,pred_type='uv')
                    pred_illum = batch["input_rgb"] / (pred_rgb + 1e-8)
                ones = torch.ones_like(batch["gt_illum"][:,:1,:,:])
                gt_illum = torch.cat([batch["gt_illum"][:,:1,:,:],ones,batch["gt_illum"][:,1:,:,:]],dim=1)

                # error metrics
                MAE_illum = float(get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"]))
                MAE_rgb = float(get_MAE(pred_rgb,batch["gt_rgb"],tensor_type="rgb",camera=self.camera,mask=batch["mask"]))
                PSNR = float(get_PSNR(pred_rgb,batch["gt_rgb"],white_level=self.white_level))

                total_mae_illum += MAE_illum * validbatch_len
                total_mae_rgb += MAE_rgb * validbatch_len
                total_psnr += PSNR * validbatch_len


            avg_loss = total_loss / total_len
            avg_mae_illum = total_mae_illum / total_len
            avg_mae_rgb = total_mae_rgb / total_len
            avg_psnr = total_psnr / total_len

            # print training log & write on tensorboard & reset vriables
            print(f'[Validation] Epoch [{epoch+1} / {self.num_epochs}] | ' \
                    f'loss: {avg_loss:.6f} | ' \
                    f'MAE_illum: {avg_mae_illum:.6f} | '\
                    f'MAE_rgb: {avg_mae_rgb:.6f} | '\
                    f'PSNR: {avg_psnr:.6f}')
            self.writer.add_scalar('validation/loss', \
                                    avg_loss, \
                                    epoch)
            self.writer.add_scalar('validation/MAE_illum', \
                                    avg_mae_illum, \
                                    epoch)
            self.writer.add_scalar('validation/MAE_rgb', \
                                    avg_mae_rgb, \
                                    epoch)
            self.writer.add_scalar('validation/PSNR', \
                                    avg_psnr, \
                                    epoch)

            # Save best U-Net model
            if avg_loss < best_unet_score:
                best_unet_score = avg_loss
                best_unet = self.unet.module.state_dict()
                print(f'Best Unet Score : {best_unet_score:.4f}')
                torch.save(best_unet, os.path.join(self.model_path, 'best.pt'))
            # # Save every 10 epoch
            # elif epoch % 10 == 9:
            #     state_dict = self.unet.module.state_dict()
            #     torch.save(state_dict, os.path.join(self.model_path, str(epoch)+'.pt'))

    def test(self):
        """Test UNet"""
        print("Test UNet!")
        self.unet.eval()

        test_loss = []
        test_MAE_img = []
        test_MAE_illum = []
        test_PSNR = []
        test_SSIM = []

        for i, (I_patch, O_patch, inputs, GTs, masks, fname) in enumerate(self.test_loader):
            if i == 46:
                continue
            I_patch = I_patch.to(self.device)
            O_patch = O_patch.to(self.device)
            inputs = inputs.to(self.device)
            GTs = GTs.to(self.device)
            masks = masks.to(self.device)

            # inference
            pred = self.unet(inputs)
            pred_patch = apply_wb(I_patch, pred, self.output_mode)
            pred_patch_rgb = pred_patch.permute(0,2,3,1).detach().cpu().numpy()

            loss = self.criterion(pred, GTs)
            test_loss.append(loss.item())

            if self.output_mode == 'illumination':
                AE = get_AE(pred.permute(0,2,3,1).detach().cpu().numpy(), 
                            GTs.permute(0,2,3,1).cpu().numpy(),
                            masks.permute(0,2,3,1).cpu().numpy())
                PSNR = get_PSNR(pred_patch_rgb, O_patch.permute(0,2,3,1).cpu().numpy(), self.white_level)

            elif self.output_mode == 'uv':
                AE_img = get_chroma_AE(pred,GTs,clip=self.camera)
                test_MAE_img.append(AE_img)

                PSNR = get_PSNR(pred_patch_rgb, O_patch.permute(0,2,3,1).cpu().numpy(), self.white_level)
                test_PSNR.append(PSNR)

                SSIM = get_SSIM(pred_patch_rgb[0], O_patch.permute(0,2,3,1).cpu().numpy()[0], self.white_level)
                test_SSIM.append(SSIM)

                mask1 = pred_patch_rgb > 1e-5
                mask1 = np.sum(mask1,axis=-1) == 3
                clipped_input_img =  np.clip(I_patch.permute(0,2,3,1).cpu().numpy(), 0, self.white_level)
                mask2 = clipped_input_img > 1e-5
                mask2 = np.sum(mask2,axis=-1) == 3
                zero_out_mask = np.logical_or(mask1,mask2)
                clipped_pred_patch_rgb = np.clip(pred_patch_rgb, 0, self.white_level)
                clipped_gt_patch_rgb= np.clip(O_patch.permute(0,2,3,1).cpu().numpy(), 0, self.white_level)

                pred_illum = clipped_input_img / (clipped_pred_patch_rgb + 1e-8)
                gt_illum = np.load(os.path.join(self.db_root,self.mode,os.path.splitext(fname[0])[0]+"_illum.npy"))
                AE_illum = get_illum_AE(pred_illum, gt_illum, zero_out_mask)
                test_MAE_illum.append(AE_illum)

            # print validation log & write on tensorboard & reset vriables
            print(f'[Test] ' \
                f'Batch [{i+1} / {len(self.test_loader)}] | ' \
                f'loss: {loss.item():.6f} | ' \
                f'MAE: {AE_illum:.6f} | ' \
                f'PSNR: {PSNR:.6f} | ' \
                f'SSIM: {SSIM:.6f}')

            if self.save_result == 'yes':
                input_rendered, output_rendered, gt_rendered = visualize(I_patch.permute(0,2,3,1).cpu()[0],pred_patch_rgb[0],O_patch.permute(0,2,3,1).cpu()[0],
                                                                         templete=self.camera, output_mode=self.output_mode, concat=False)
                cv2.imwrite(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_input.png'), 
                            cv2.cvtColor(input_rendered, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+f'_output_{AE_illum:.2f}_{PSNR:.2f}.png'), 
                            cv2.cvtColor(output_rendered, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_gt.png'), 
                            cv2.cvtColor(gt_rendered, cv2.COLOR_RGB2BGR))

                # additional save for uv-distribution visualization & error map
                np.save(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_input'), clipped_input_img)
                np.save(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_output'), clipped_pred_patch_rgb)
                np.save(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_gt'), clipped_gt_patch_rgb)
                np.save(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_illum_gt'), gt_illum)
                np.save(os.path.join(self.result_path,str(i)+'_'+os.path.splitext(fname[0])[0]+'_illum_output'), pred_illum)

        print("LOSS :", np.mean(test_loss), np.median(test_loss), np.max(test_loss))
        print("MAE_ILLUM :", np.mean(test_MAE_illum), np.median(test_MAE_illum), np.max(test_MAE_illum))
        print("MAE_IMG :", np.mean(test_MAE_img), np.median(test_MAE_img), np.max(test_MAE_img))
        print("PSNR :", np.mean(test_PSNR), np.median(test_PSNR), np.max(test_PSNR))
        print("SSIM :", np.mean(test_SSIM), np.median(test_SSIM), np.max(test_SSIM))