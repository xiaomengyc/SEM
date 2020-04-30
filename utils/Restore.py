import os
import torch
import pdb

__all__ = ['restore']


def find_lasted_save_checkpoint(restore_dir):
    filelist = os.listdir(restore_dir)
    filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('.pth.tar')]
    if len(filelist) > 0:
        filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
        snapshot = os.path.join(restore_dir, filelist[0])
    else:
        snapshot = None
    return snapshot

def full_restore(args, model, snapshot):
    # if os.path.isfile(args.restore_from) and ('.pth' in args.restore_from):
    #     snapshot = args.restore_from
    # else:
    #     snapshot = find_lasted_save_checkpoint(args.snapshot_dir)

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)

        if args.resume == "True":
            if 'epoch' in list(checkpoint.keys()):
                args.current_epoch = checkpoint['epoch'] + 1
            if 'global_counter' in list(checkpoint.keys()):
                args.global_counter = checkpoint['global_counter'] + 1


        try:
            model.load_state_dict(checkpoint["state_dict"])
        except KeyError:
            model.load_state_dict(checkpoint)



def partial_restore(args, model, snapshot):
    # if os.path.isfile(args.restore_from) and ('.pth' in args.restore_from):
    #     snapshot = args.restore_from
    # else:
    #     snapshot = find_lasted_save_checkpoint(args.snapshot_dir)

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            if args.resume == "True":
                args.current_epoch = checkpoint['epoch'] + 1
                args.global_counter = checkpoint['global_counter'] + 1

            model_dict = model.state_dict()
            model_keys = list(model_dict.keys())

            if 'state_dict' in list(checkpoint.keys()):
                checkpoint = checkpoint['state_dict']

            # new_dict = {k:v for k,v in checkpoint['state_dict'].items() if (k in model_keys) and (v.size() == model_dict[k].size())}
            new_dict = {k:v for k,v in checkpoint.items() if (k in model_keys) and (v.size() == model_dict[k].size())}
            print('=> The following parameters cannot be reloaded!:')
            print([k for k in model_keys if k not in list(new_dict.keys())])
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
            # pdb.set_trace()

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(snapshot, args.current_epoch))
        except KeyError:
            print("=> Loading pre-trained values failed.")
            raise

    else:
        print("=> Warning: no checkpoint found at '{}'".format(snapshot))


def restore(args, model):
    if os.path.isfile(args.restore_from): # and ('.pth' in args.restore_from):
        snapshot = args.restore_from
    else:
        snapshot = find_lasted_save_checkpoint(args.snapshot_dir)

    if snapshot is not None and os.path.isfile(snapshot):
        try:
            full_restore(args, model, snapshot)
        except RuntimeError:
            print("=> Full restore failed, try partial restore.")
            partial_restore(args, model, snapshot)
    else:
        print("=> Warning: no checkpoint found")

