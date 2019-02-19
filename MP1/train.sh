method=cg # cg or pr
epochs=2 # need experiment
larger_param=False # True: large parameters; False: small
full_batch=True
no_restart=False
python3 mp1.py --method $method --epochs $epochs --larger_param $larger_param --full_batch $full_batch --no_restart $no_restart
