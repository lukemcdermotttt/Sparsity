python prune.py --model 'resnet-18' --prune_method 'delta_pruning' --save_dir './weights/rn18_delta/' --num_epochs 100
#python prune.py --model 'resnet-18' --prune_method 'lottery_ticket_rewinding' --save_dir './weights/rn18_ltr/'
#python prune.py --model 'resnet-18' --prune_method 'early_training_rewinding' --save_dir './weights/rn18_k=5/'
#python prune.py --model 'mlp_mixer' --prune_method 'early_training_rewinding' --save_dir './weights/mlp_k=5'