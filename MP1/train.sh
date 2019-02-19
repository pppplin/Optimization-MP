method=cg # cg or pr
epochs=10
larger_param=False
reg=False
restart=False
python3 mp1.py --method $method --epochs $epochs --larger_param $larger_param --reg $reg --restart $restart
