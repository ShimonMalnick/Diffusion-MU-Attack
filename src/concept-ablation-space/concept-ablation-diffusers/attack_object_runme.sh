methods=('EraseDiff' 'ESD' 'FMN' 'Salun' 'Scissorhands' 'SPM' 'UCE')
object_names=('Parachute' 'Garbage_Truck' 'Church' 'Tench')
for object_name in "${object_names[@]}"; do
    for method in "${methods[@]}"; do
        echo "Inverting $method ...."
        python concept-ablation-diffusers/attack_tamed_model_variable_object.py --method $method --prompt $object_name
    done
done
# for method in "${methods[@]}"; do
#     echo "Inverting $method ...."
#     python concept-ablation-diffusers/attack_tamed_model_variable_object.py --method $method
# done
