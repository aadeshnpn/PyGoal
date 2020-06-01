import torch
from mario_temporal import (
    MarioPolicyNetwork, Generator, MarioEnvironment,
    ResNet50Bottom)
from utils import (
    prepare_numpy, create_trace_flloat,
    create_trace_skeleton, trace_accumulator,
    prepare_input
    )
import imageio
import numpy as np
from array2gif import write_gif
from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict
import torchvision.models as models
import skvideo.io



def make_gif(x, filename):
    print(len(x), x[0].shape)
    # print(np.max(x[0]))
    # print(x[0])
    # write_gif(x, '0.gif', fps=5)
    # x = x.astype(np.uint8)
    skvideo.io.vwrite("out.mp4", x)
    # with imageio.get_writer(
    #         filename, mode='I', duration=1 / 30) as writer:

    #     writer.append_data((x[:, :, 0] * 255).astype(np.uint8))


def render(policy, embedding_net, device):
    from torchvision import transforms
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
    env = MarioEnvironment()
    s = env.reset()
    images = [s]
    s = s.reshape(s.shape[1], s.shape[2], s.shape[0])
    # print(s.shape)
    s = trans(s)
    # temp = s[..., np.newaxis] * np.ones(3)
    # temp = temp.squeeze()
    # temp = temp.reshape(temp.shape[1], temp.shape[0], temp.shape[2])
    # print(temp.shape)
    coin = 0
    for _ in range(60):
        # env.render()
        # s = np.reshape(s, (s.shape[0]*s.shape[1]*s.shape[2]))
        input_state = prepare_input(s)
        input_state = embedding_net(input_state)

        action_dist, action = policy(input_state)
        action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
        s_prime, r, t, coins = env.step(action)
        coin += coins
        if t:
            break
        s = s_prime
        images.append(s)
        s = s.reshape(s.shape[1], s.shape[2], s.shape[0])
        # s = s.reshape(s.shape[0] * s.shape[1] * s.shape[2])
        s = trans(s)

        # temp = s[..., np.newaxis] * np.ones(3)
        # temp = temp.squeeze()
        # temp = temp.reshape(temp.shape[1], temp.shape[0], temp.shape[2])

    # Create gifs
    print('total coins', coin)
    make_gif(images, '0.gif')


def recognition(trace, keys):
    goalspec = 'F P_[C][1,none,<=]'
    # parse the formula
    parser = LTLfGParser()

    # Define goal formula/specification
    parsed_formula = parser(goalspec)

    # Change list of trace to set
    traceset = trace.copy()
    akey = list(traceset.keys())[0]
    # print('recognizer', traceset)
    # Loop through the trace to find the shortest best trace
    for i in range(0, len(traceset[akey])+1):
        t = create_trace_flloat(traceset, i, keys)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, i

    return result, i

def slice_trace(j, trace, keys):
    for k in keys:
        trace[k] = trace[k][j:]
    return trace


def temp_fn(gamma, ret, trajectory):
    for i in reversed(range(len(trajectory))):
        state = trajectory[i]
        trajectory[i] = (state, ret)
        ret = ret * gamma
    return trajectory

def calculate_returns(trajectory, gamma, trace, keys):
    # ret = finalrwd
    # print(trajectory)
    tlen = len(trajectory)
    result, j = recognition(trace, keys)
    if result is False:
        ret = -1
        return temp_fn(gamma, ret, trajectory)
    else:
        if result is True and  j+1 >= tlen:
            ret = +1
            return temp_fn(gamma, ret, trajectory)
        else:
            ret = +1
            traj = temp_fn(gamma, ret, trajectory[:j])
            traj += calculate_returns(trajectory[j:], gamma, slice_trace(j+1, trace, keys), keys)
            return traj
    # print(trajectory)
    # return trajectory


def test_return():
    keys = ['C']
    s = 0
    trace = create_trace_skeleton([s], keys)
    rollout = list(range(20))
    for _ in range(20):
        s_prime = np.random.choice([0, 1], p=[0.95,0.05])
        trace_accumulator(trace, [s_prime], keys)
    print(trace)
    # print(rollout)
    rollout = calculate_returns(rollout, 0.9, trace, keys)
    print(rollout)


def main():
    # embedded = Generator()
    # embedded.load_state_dict(torch.load("embedded.pt"))
    res18_model = models.vgg11(pretrained=True)
    embedded = ResNet50Bottom(res18_model)
    for para in res18_model.parameters():
        para.requires_grad = False
    policy = MarioPolicyNetwork()
    policy.load_state_dict(torch.load("policy.pt"))
    render(policy, embedded, 'cpu')


if __name__ == '__main__':
    main()
    # test_return()
# torch numpy gym tqdm matplotlib opencv-python nes_py gym-super-mario-bros baselines