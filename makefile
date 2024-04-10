bs_infer:
	python ./batch_infer.py --model_name_or_path /home/ring/Documents/workspace/modules/tinyllama-110M

infer:
	python ./infer.py --model_name_or_path /home/ring/Documents/workspace/modules/tinyllama-110M

bench:
	python ./benchmark.py --model_path ./llama-zero
