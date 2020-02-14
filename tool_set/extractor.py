'''
以spp为基础，对模型进行蒸馏。由于时间紧迫，目前还不能逐层蒸馏，只能一次性批量性的减小层数
'''
import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *

ONNX_EXPORT = False


def create_modules(module_defs, scale, fd, minc=3, maxc=32):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = []
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    layer_idx = 0

    for i, mdef in enumerate(module_defs):

        if mdef['type'] == 'convolutional':
            modules = "layer%d"%layer_idx
            cmdf = "self.{} = self.CBL(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, bn={}, leaky={})\n"
            bn = int(mdef['batch_normalize'])
            if module_defs[i+1]['type'] != 'yolo':
                filters = int(min(max(minc, int(mdef['filters'])/scale),maxc))
            else:
                filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            fd.write(cmdf.format(modules, output_filters[-1], filters, kernel_size, int(mdef['stride']), pad, bn,
                                 mdef['activation'] == 'leaky'))
            layer_idx += 1

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            modules = "maxpool%d"%layer_idx
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                fd.write("self.{} = nn.Sequential(nnn.ZeroPad2d((0, 1, 0, 1))," % modules)
                fd.write("nn.MaxPool2d(kernel_size={}, stride={}, padding={})".format(i, kernel_size,
                                                                                        stride, int(
                        (kernel_size - 1) // 2)))
                fd.write(")\n")
            else:
                fd.write("self.{} = nn.MaxPool2d(kernel_size={}, stride={}, padding={})\n".format(modules, kernel_size,
                                                                                                       stride, int(
                        (kernel_size - 1) // 2)))
            layer_idx += 1

        elif mdef['type'] == 'upsample':
            modules = 'upSample%d'%layer_idx
            fd.write("self.{} = nn.Upsample(scale_factor={}, mode='nearest')\n".format(modules, int(mdef['stride'])))
            layer_idx += 1

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            modules = "yolo"

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs

class Darknet():
    # YOLOv3 object detection model

    def __init__(self, cfg, scale, img_size=(416, 416), arc='default'):
        self.fd = open("distillation.txt", "w")
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, scale, self.fd)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self):
        self.fd.write("========>forward\n")
        layer_outputs = []
        output = []
        x = "X"
        yolo_i = 0
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                self.fd.write("{} = self.{}({})\n".format(module, module, x))
                x = module
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    x = "brandch%d"%i
                    self.fd.write("{} = torch.cat({}, 1)\n".format(x, [layer_outputs[i] for i in layers]))
            elif mtype == 'shortcut':
                _ = "shortcut%d"%i
                self.fd.write("{} = {} + {}\n".format(_, x, layer_outputs[int(mdef['from'])]))
                x = _
                #x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                self.fd.write("yolo{} = {}\n".format(yolo_i, x))
                x = "yolo%d"%yolo_i
                yolo_i += 1
                #x = module(x, img_size)
                #output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        self.fd.close()

if __name__ == "__main__":
    Darknet("cfg/yolov3-spp.cfg", scale=16).forward()