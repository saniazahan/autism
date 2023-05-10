stream = 'skel' # skel  rgb video video_skel

# directory containing frames
#train_label_dir = './data/train_file_names_v5.pkl'
#test_label_dir = './data/test_file_names_v5.pkl'
seed = 1
seed_type = 'block' # block  random

test_type = 'Aug' # NoAug Aug
if seed_type == 'random':
    train_data_dir = './data/skeleton/train_data_skel_{}.npy'.format(seed)
    train_label_dir = './data/skeleton/train_labels_skel_{}.pkl'.format(seed)
    test_data_dir = './data/skeleton/test_data_skel_{}.npy'.format(seed)
    test_label_dir = './data/skeleton/test_labels_skel_{}.pkl'.format(seed)
else:
    train_data_dir = './data/skeleton/train_data_block_{}.npy'.format(seed)
    train_label_dir = './data/skeleton/train_labels_block_{}.pkl'.format(seed)
    test_data_dir = './data/skeleton/test_data_block_{}.npy'.format(seed)
    test_label_dir = './data/skeleton/test_labels_block_{}.pkl'.format(seed)

normalization = False
 
GaitEnergy = True   #for skepxel

video_stream = False

GCN_stream = True  #for skeleton stream

use_vit_for_cluster_loss_only = False

cluster_distance_loss = False

temporal_aug = 6

class_embed_type = 'euclidean' # euclidean hyperbolic
distance_type =  'euclidean' # euclidean hyperbolic

Trajectory = False

cross_validation = False

folds = 10

hyp_c = 0.1  # hyperbolic c, "0" enables sphere mode
tau = 0.2  # cross-entropy temperature

# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './data/rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames = 64#112
if '_v' in train_label_dir:
    num_clip = 3
else:
    num_clip = 6

num_class = 2
# beginning frames of the 10 segments
keep_prob = 0.0
block_size = 7


# input data dims;
C, H, W = 3,224,224
# image resizing dims;
input_resize = 455, 256



# output dimension of I3D backbone
feature_dim = 1024


inference = False

states_path = "./exps/skepxel_1/weights/Vit_b_weights.pt"

x3d_state_path = './exps/vid_baseline_random/weights/x3d_weights.pt'