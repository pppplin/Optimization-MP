method=cg # cg or pr
epochs=1000
larger_param=False
reg=False
restart=False
python maml.py --method $method --epochs $epochs --larger_param $larger_param --reg $reg --restart $restart
