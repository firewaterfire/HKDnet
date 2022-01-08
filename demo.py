import torch
from net import Student_sr, Student_fu
from option import args
import os
import numpy as np
import numpy
from torch.autograd import Variable
from torchvision import transforms
from imageio import imread, imsave

def load_model():
    fusion_part = Student_fu(args).cuda()
    sr_part = Student_sr(args).cuda()
    model_dir = './checkpoint/2x/fu_srnet.pth.tar'
    checkpoint = torch.load(model_dir)
    fusion_part.load_state_dict(checkpoint['Sec_Fusionnet_state_dict'])
    sr_part.load_state_dict(checkpoint['srnet_state_dict'])

    para_fu = sum([np.prod(list(p.size())) for p in fusion_part.parameters()])
    para_sr = sum([np.prod(list(p.size())) for p in sr_part.parameters()])
    para = para_fu + para_sr

    type_size = 4
    print('Model {} params: {:4f}M'.format("Student_Network", para * type_size / 1000 / 1000))

    fusion_part.eval()
    fusion_part.cuda()
    sr_part.eval()
    sr_part.cuda()

    return fusion_part, sr_part

def _generate_fusion_image(model, img1, img2):
	en_r = model.encoder_i(img1)
	en_v = model.encoder_v(img2)
	f = model.fusion(en_r, en_v)
	img_fusion_fe = model.decoder(f)
	# img_fusion = model.result(img_fusion_fe[0])
	return img_fusion_fe

def _generate_sr_image(model, lr_fusion_image):
    sr_image_fusion = model(lr_fusion_image[0])
    sr_image = sr_image_fusion[0]
    return sr_image

def save_images(path, data):
    if data.shape[0] == 1:
        data = data.reshape([data.shape[1], data.shape[2]])
    imsave(path, data)

def run_demo(fusion_part, sr_part, lr_infrared_path, lr_visible_path, output_path_root, index, mode):
    lr_ir_img = get_test_images(lr_infrared_path, mode=mode)
    lr_vis_img = get_test_images(lr_visible_path, mode=mode)
    if args.cuda:
        lr_ir_img = lr_ir_img.cuda()
        lr_vis_img = lr_vis_img.cuda()

    lr_ir_img = Variable(lr_ir_img, requires_grad=False)
    lr_vis_img = Variable(lr_vis_img, requires_grad=False)

    img_fusion = _generate_fusion_image(fusion_part, lr_ir_img, lr_vis_img)

    sr_image_fusion = _generate_sr_image(sr_part, img_fusion)
    sr_image_fusion = torch.squeeze(sr_image_fusion)
    sr_image_fusion = sr_image_fusion.cpu().mul_(255).numpy()

    file_name = str(index) + '.png'
    output_path = output_path_root + file_name
    save_images(output_path, sr_image_fusion)
    print(output_path)

def get_test_images(paths,  mode='L'):
	ImageToTensor = transforms.Compose([transforms.ToTensor()])
	if isinstance(paths, str):
		paths = [paths]
	images = []
	for path in paths:
		image = imread(path)
		if mode == 'L':
			image = np.reshape(image, [1, image.shape[0], image.shape[1]])
			amin, amax = image.min(), image.max()  #
			image = (image - amin) / (amax - amin)
		else:
			image = ImageToTensor(image).float().numpy()*255
	images.append(image)
	images = np.stack(images, axis=0)
	images = torch.from_numpy(images).float()
	return images

def main():
    test_path_lr = "./sourceimages/2x/"
    output_path = './outputs/2x/'
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    mode = "L"
    with torch.no_grad():
        model = load_model()
        fusion_part = model[0]
        sr_part = model[1]
        for i in range(0,6):
            index = i + 1
            lr_infrared_path = test_path_lr + 'IR' + str(index) + '.png'
            lr_visible_path = test_path_lr + 'VIS' + str(index) + '.png'
            run_demo(fusion_part, sr_part, lr_infrared_path, lr_visible_path, output_path, index, mode)

    print('Done......')

if __name__ == '__main__':
    main()
