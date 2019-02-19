method=pr # cg or pr
epochs=100 # need experiment
larger_param=False # True: large parameters; False: small
reg=False #This is for hessian TODO
no_restart=True
python3 mp1.py --method $method --epochs $epochs --larger_param $larger_param --reg $reg --no_restart $no_restart
