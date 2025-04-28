model_path_list=("xx/ReTool-Qwen-32B" "xx/ReTool-DeepSeek-R1-Distill-Qwen-32B")
dataset_list=("AIME24" "AIME25")
prompt_template_path="prompt_template.json"

num_model=${#model_path_list[@]}
num_dataset=${#dataset_list[@]}
for ((i=0;i<$num_model;i++)) do
{
    for ((j=0; j<$num_dataset; j++)); do
    { 
        MODEL_PATH=${model_path_list[$i]}
        DATA_NAME=${dataset_list[$j]}
        SRC_PATH="results"

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        python eval.py \
            --data_name ${DATA_NAME} \
            --target_path ${SRC_PATH} \
            --model_name_or_path ${MODEL_PATH} \
            --max_tokens 16384 \
            --paralle_size 8 \
            --n 32 \
            --prompt_template ${prompt_template_path} \
            --prompt retool \
            --exe_code &
        wait
    }
    done

}
done