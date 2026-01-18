import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='diffu', help='Model to train (diffu, layers, consis, cond_dist)')
parser.add_argument('-c', '--config', default='config_dataset2.json', help='Config file with training parameters')
parser.add_argument('-n', '--name', default='test', help='job name')
parser.add_argument("--resubmit", default = False, action='store_true')
parser.add_argument("--constraint", default = "a100|v100|p100" , help='gpu resources')
parser.add_argument("--memory", default = 16000 , help='RAM')
parser.add_argument("--extra_args", default = "" , help='Extra arguments to pass to training script')
parser.add_argument("--n-nodes", type=int, default=1, help='Number of nodes for DDP training')
parser.add_argument("--gpus-per-node", type=int, default=1, help='Number of GPUs per node for DDP training')
parser.add_argument("--checkpoint", default="../models", help='Checkpoint folder')

flags = parser.parse_args()

if(flags.name[-1] =="/"): flags.name = flags.name[:-1]
print(flags.name)

base_dir = r"\/work1\/cms_mlsim\/oamram\/CaloDiffusion\/slurm\/"
checkpoint_dir = flags.checkpoint.replace("/", "\/")

if(flags.model in ['diffu', 'layers', 'consis', 'cond_dist']):
    if(not os.path.exists(flags.name)): os.system("mkdir %s" % flags.name)
    cfg_loc = flags.name + "/config.json"
    
    # Use DDP script if multi-node or multi-GPU
    use_ddp = flags.n_nodes > 1 or flags.gpus_per_node > 1
    
    if use_ddp:
        script_template = "diffu_train_ddp.sh"
        script_loc = flags.name + "/diffu_train_ddp.sh"
    else:
        script_template = "diffu_train.sh"
        script_loc = flags.name + "/diffu_train.sh"
    
    if(not flags.resubmit):
        os.system("cp %s %s" % (flags.config, cfg_loc))
        os.system("cp %s %s" % (script_template, script_loc))
        os.system("sed -i 's/JOB_NAME/%s/' %s" % (flags.name, script_loc))
        os.system("sed -i 's/JOB_OUT/%s/' %s" % (flags.name, script_loc))
        os.system("sed -i 's/MODELTYPE/%s/' %s" % (flags.model, script_loc))
        os.system("sed -i 's/CONSTRAINT/%s/' %s" % (flags.constraint, script_loc))
        os.system("sed -i 's/MEMORY/%s/' %s" % (flags.memory, script_loc))
        os.system("sed -i 's/CONFIG/%s/' %s" % (base_dir + flags.name + "\/config.json", script_loc) )
        os.system("sed -i 's/CHECKPOINT/%s/' %s" % (checkpoint_dir, script_loc) )
        os.system("sed -i 's/EXTRAARGS/%s/' %s" % (flags.extra_args, script_loc) )
        
        # Add DDP-specific substitutions
        if use_ddp:
            os.system("sed -i 's/NNODES/%s/' %s" % (flags.n_nodes, script_loc))
            os.system("sed -i 's/NGPUS/%s/' %s" % (flags.gpus_per_node, script_loc))
        
    os.system("sbatch %s" % script_loc)
else:
    print("Unrecognized model %s" % flags.model)

