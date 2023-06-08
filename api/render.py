import numpy as np
from mayavi import mlab
import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
import math

from net import Net

model = Net()
model.load_state_dict(torch.load('mnist.pth')) #loads model. upload this file bc that's where the model is stored?
model.eval().to('cpu')

def convolute(blur_kernel, cross_section):
    new_pixel = 0
    for i in range(len(blur_kernel)):
        for j in range(len(blur_kernel[i])):
            new_pixel += blur_kernel[i][j] * cross_section[i][j]

    return new_pixel

def rendercnn():
    # Define Gaussian Blur Kernel
    blur_kernel = [
        [0.025, 0.125, 0.025], 
        [0.125, 1.000, 0.125], 
        [0.025, 0.125, 0.025]
    ]

    # Offset array
    kernel_offsets = [
        [-1, -1], [-1, 0], [-1, 1], 
        [0, -1], [0, 0], [0, 1], 
        [1, -1], [1, 0], [1, 1], 
    ]

    with open('matrix.json', 'r') as file:
        matrix_data = json.load(file)
        modified_matrix_data = matrix_data.copy()
    x = [[matrix_data]] # formatting - extra wrapping 
    x_prime = [[modified_matrix_data]] # create copy for convolution
    print(f'This is x: {x}')

    # Padding layer to exaggerate features
    for i in range(len(x[0][0])):
        for j in range(len(x[0][0][i])):
            new_i = i + 1
            new_j = j

            # Left shift padding cause I said so 
            try: 
                new_value = x_prime[0][0][new_i][new_j]
                if(new_value > 0):
                    x_prime[0][0][i][j] = 1
            except:
                print("Ignore error lmao")

    for i in range(len(x[0][0])):
        for j in range(len(x[0][0][i])):

            # Initialize image cross-section for convolution
            cross_section = [
                [0, 0, 0], 
                [0, 0, 0], 
                [0, 0, 0]
            ]

            # Construct the cross_section matrix and then apply blur
            for offset in kernel_offsets:
                new_i = i + offset[0]
                new_j = j + offset[1]
                try: 
                    cross_section[offset[0] + 1][offset[1] + 1] = x_prime[0][0][new_i][new_j]
                except:
                    print("Ignore error lmao")

            # Apply blur
            new_value = convolute(blur_kernel, cross_section)

            if new_value < 0.05:
                new_value = 0.15 # negative thresholding for more mimicking
            
            x[0][0][i][j] = new_value

        #print(x[0][0][i])
        print(x[0][0][i])
    print(f'This is pre blur: {x_prime}')
    print(f'This is x post blur: {x}')

    input = list()
    conv1 = list()
    conv2 = list()
    fc1 = list()
    output = list()
    x = torch.Tensor(list(x))

    # Using MNIST dataset for testing purposes - uncomment below
    # from torchvision import datasets, transforms
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # x, y = dataset1[7] #go through images of dataset
    # x = x.unsqueeze(0)

    # Run output through model
    input.append(np.abs(x[0].detach().numpy()))
    x = model.conv1(x)
    x = F.relu(x)
    conv1.append(np.abs(x[0].detach().numpy()))
    x = model.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    conv2.append(np.abs(x[0].detach().numpy()))
    x = torch.flatten(x, 1)
    x = model.fc1(x)
    x = F.relu(x)
    fc1.append(np.abs(x[0].detach().numpy()))
    x = model.fc2(x)
    x = F.log_softmax(x, dim=1)
    output.append(np.abs(x[0].detach().numpy()))
    input = np.array(input)
    conv1 = np.array(conv1)
    conv2 = np.array(conv2)
    fc1 = np.array(fc1)
    output = np.array(output)

    act_input = input[0][0]
    act_conv1 = conv1[0]
    act_conv2 = conv2[0]
    act_fc1 = fc1[0]
    act_out = output[0]

    fig = mlab.figure(bgcolor=(13 / 255, 21 / 255, 44 / 255), size=(1920, 1080))

    img_input = mlab.imshow(act_input, colormap='gray', interpolate=False, extent=[-27, 27, -27, 27, 1, 1])
    img_input.actor.position = [0, 0, 0]
    img_input.actor.orientation = [0, 90, 90]
    img_input.actor.force_opaque = True

    # Conv1
    img_conv1 = list()
    for row in range(6):
        for col in range(6):
            img = mlab.imshow(act_conv1[row * 6 + col], colormap='gray', interpolate=False)
            img.actor.position = [(row - 2.5) * 26, 50, (col - 2.5) * 26]
            img.actor.orientation = [0, 90, 90]
            img.actor.force_opaque = True
            img_conv1.append(img)

    # Conv2
    img_conv2 = list()
    for row in range(8):
        for col in range(8):
            img = mlab.imshow(act_conv2[row * 8 + col], colormap='gray', interpolate=False)
            img.actor.position = [(row - 3.5) * 12, 100, (col - 3.5) * 12]
            img.actor.orientation = [0, 90, 90]
            img.actor.force_opaque = True
            img_conv2.append(img)

    # Create the points
    conv2_x = list()
    conv2_y = list()
    for row in range(8):
        for col in range(8):
            x, y = np.indices((12, 12))
            x = x.ravel() + (row - 4) * 12
            y = y.ravel() + (col - 4) * 12
            conv2_x.append(x)
            conv2_y.append(y)
    conv2_x = np.hstack(conv2_x)
    conv2_y = np.hstack(conv2_y)
    conv2_z = np.ones_like(conv2_x) * 100

    fc1_x, fc1_y = np.indices((12, 12))
    fc1_x = fc1_x.ravel() * 2 - 12
    fc1_y = fc1_y.ravel() * 2 - 12
    fc1_z = np.ones_like(fc1_x) * 150 + np.random.rand(*fc1_x.shape) * 10

    out_x = np.arange(10)
    out_x = (out_x.ravel() - 5) * 3
    out_y = np.zeros_like(out_x)
    out_z = np.ones_like(out_y) + 200

    x = np.hstack([conv2_x, fc1_x, out_x])
    y = np.hstack([conv2_y, fc1_y, out_y])
    z = np.hstack([conv2_z, fc1_z, out_z])
    s = np.hstack([
        act_conv2.ravel() / np.max(act_conv2),
        act_fc1 / np.max(act_fc1),
        1 - (act_out / np.max(act_out)),
    ])
    acts = mlab.points3d(x[len(act_conv2.ravel()):], z[len(act_conv2.ravel()):], y[len(act_conv2.ravel()):], s[len(act_conv2.ravel()):], mode='cube', scale_factor=1, scale_mode='none', colormap='gray')

    # Connections between the layers
    fc1 = model.fc1.weight.detach().numpy().T
    out = model.fc2.weight.detach().numpy().T
    fr_conv2, to_fc1 = (np.abs(fc1) > 0.08).nonzero()
    fr_fc1, to_out = (np.abs(out) > 0.2).nonzero()
    to_fc1 += len(conv2_x)
    fr_fc1 += len(conv2_x)
    to_out += len(conv2_x) + len(fc1_x)
    c = np.vstack((
        np.hstack((fr_conv2, fr_fc1)),
        np.hstack((to_fc1, to_out)),
    )).T

    src = mlab.pipeline.scalar_scatter(x, z, y, s)
    src.mlab_source.dataset.lines = np.vstack((
        np.hstack((fr_conv2, fr_fc1)),
        np.hstack((to_fc1, to_out)),
    )).T
    src.update()
    lines = mlab.pipeline.stripper(src)
    connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

    # Text
    mlab.text3d(x=-14.5, y=200.5, z=-2, text='0')
    mlab.text3d(x=-11.5, y=200.5, z=-2, text='1')
    mlab.text3d(x=-8.5, y=200.5, z=-2, text='2')
    mlab.text3d(x=-5.5, y=200.5, z=-2, text='3')
    mlab.text3d(x=-2.5, y=200.5, z=-2, text='4')
    mlab.text3d(x=0.5, y=200.5, z=-2, text='5')
    mlab.text3d(x=3.5, y=200.5, z=-2, text='6')
    mlab.text3d(x=6.5, y=200.5, z=-2, text='7')
    mlab.text3d(x=9.5, y=200.5, z=-2, text='8')
    mlab.text3d(x=12.5, y=200.5, z=-2, text='9')

    mlab.view(0, 90, 400, [0, 100, 0])

    # Update the data and view
    @mlab.animate(delay=100, ui=True)
    def anim():
        img_input.mlab_source.scalars = input[0][0]
        for img, a in zip(img_conv1, conv1[0]):
            img.mlab_source.scalars = a
        for img, a in zip(img_conv2, conv2[0]): #pictures updating
            img.mlab_source.scalars = a
        act_fc1 = fc1[0]
        act_out = output[0]
        s = np.hstack((
            act_conv2.ravel() / act_conv2.max(),
            act_fc1 / act_fc1.max(),
            1 - (act_out / act_out.max())
        ))
        acts.mlab_source.scalars = s[len(act_conv2.ravel()):]
        connections.mlab_source.scalars = s
        mlab.view(azimuth=(0 / 2) % 360, elevation=80, distance=300, focalpoint=[0, 100, 0], reset_roll=False)
        #mlab.savefig(f'/l/vanvlm1/scns/frame{frame:04d}.png') #for saving frame by frame of the animation to make a GIF or something out of later
        yield

    anim()
    mlab.show()

def main():
    rendercnn() #matrix input here
    
if __name__ == "__main__": 
    main()

# x = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
#         [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0]]