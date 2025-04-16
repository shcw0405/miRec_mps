import os
import time
    
def get_exp_name(args, save=True, exp='e1'):
    """
    根据不同模型类型生成定制的实验名称
    
    参数:
        args: 包含实验参数的对象
        save: 是否需要保存模型
        exp: 实验标识
    """

    # 基础参数（所有模型都需要的）
    # 智能格式化weight_decay，去掉无意义的小数点
    if float(args.weight_decay).is_integer():
        wdc_str = str(int(args.weight_decay))
    else:
        wdc_str = str(args.weight_decay)
    base_params = [
        args.dataset,
        args.model_type,
        'dropout'+str(args.dropout),
        'wdc'+wdc_str
    ]
    
    # 根据模型类型添加特定参数
    model_specific_params = []
    
    if args.model_type in ['ComiRec-SA', 'ComiRec-DR', 'MIND']:
        # ComiRec系列模型特有参数
        model_specific_params.extend([
            'd'+str(args.hidden_size),
            'in'+str(args.interest_num),
            'top'+str(args.topN)
        ])
        
    elif args.model_type in ['GRU4Rec', 'SASRec']:
        # GRU4Rec,SASRec 模型特有参数
        model_specific_params.extend([
            'd'+str(args.hidden_size),
            'top'+str(args.topN)
        ])

    elif args.model_type in ['REMI','REMIuseremb']:
        model_specific_params.extend([
            'd'+str(args.hidden_size),
            'top'+str(args.topN),
            'rbeta'+str(args.rbeta),
            'rlambda'+str(args.rlambda)
        ])
        
    
    # 合并所有参数
    all_params = base_params + model_specific_params
        
    # 创建基本名称
    para_name = '_'.join(all_params)
    
    # 使用传入的exp或args中的exp
    exp_id = exp if exp != 'e1' or not hasattr(args, 'exp') else args.exp
    base_name = para_name + '_' + exp_id
    
    # 处理重名情况
    exp_path = 'best_model/' + base_name
    if os.path.exists(exp_path) and save:
        timestamp = time.strftime("_%m%d_%H%M", time.localtime())
        exp_name = base_name + timestamp
        
        i = 1
        while os.path.exists('best_model/' + exp_name):
            exp_name = base_name + timestamp + f"_{i}"
            i += 1
    else:
        exp_name = base_name
    
    # 创建目录
    if save and not os.path.exists('best_model/' + exp_name):
        os.makedirs('best_model/' + exp_name, exist_ok=True)
        
    return exp_name