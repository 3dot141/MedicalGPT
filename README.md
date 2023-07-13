#Area/AI 

参考 [MedicalGPT](https://github.com/3dot141/MedicalGPT/blob/main/home.md) 的方案。实现

- 第一阶段：PT(Continue PreTraining) 增量预训练，在海量领域文档数据上二次预训练 GPT 模型，以注入领域知识
- 第二阶段：SFT(Supervised Fine-tuning) 有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图
- 第三阶段：RM(Reward Model) 奖励模型建模，构造人类偏好排序数据集，训练奖励模型，用来对齐人类偏好，主要是 "HHH" 原则，具体是 "helpful, honest, harmless"
- 第四阶段：RL(Reinforcement Learning) 基于人类反馈的强化学习 (RLHF)，用奖励模型来训练 SFT 模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本

## 准备阶段

准备微调环境，这里直接用 

- [Colab 使用介绍](Colab%20使用介绍.md)
- Google Drive 的方案 
	- 别忘了调成 [土耳其](../Action/Google-土耳其换区指南.md) 区，可以省一大笔钱。

## PT 阶段

### 参数说明

以下提供三种参数，模型参数、数据集参数、Peft 参数

> Peft 是对 lora 的实现。

```python file:ModelArgs
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
	    default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")

```

```python file:DataArgs

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
```

```python file:TrainArgs
@dataclass
class PeftArguments(TrainingArguments):
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)

    """
    Parameters:
        output_dir (`str`):
        overwrite_output_dir (`bool`, *optional*, defaults to `False`):
        do_train (`bool`, *optional*, defaults to `False`):
        do_eval (`bool`, *optional*):
        do_predict (`bool`, *optional*, defaults to `False`):
        evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
        prediction_loss_only (`bool`, *optional*, defaults to `False`):
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
        eval_accumulation_steps (`int`, *optional*):
        eval_delay (`float`, *optional*):
        learning_rate (`float`, *optional*, defaults to 5e-5):
        weight_decay (`float`, *optional*, defaults to 0):
        adam_beta1 (`float`, *optional*, defaults to 0.9):
        adam_beta2 (`float`, *optional*, defaults to 0.999):
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
        max_grad_norm (`float`, *optional*, defaults to 1.0):
        num_train_epochs(`float`, *optional*, defaults to 3.0):
        max_steps (`int`, *optional*, defaults to -1):
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
        warmup_ratio (`float`, *optional*, defaults to 0.0):
        warmup_steps (`int`, *optional*, defaults to 0):
        log_level (`str`, *optional*, defaults to `passive`):
        log_level_replica (`str`, *optional*, defaults to `"warning"`):
        log_on_each_node (`bool`, *optional*, defaults to `True`):
        logging_dir (`str`, *optional*):
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
        logging_first_step (`bool`, *optional*, defaults to `False`):
        logging_steps (`int` or `float`, *optional*, defaults to 500):
        logging_nan_inf_filter (`bool`, *optional*, defaults to `True`):
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
        save_steps (`int` or `float`, *optional*, defaults to 500):
        save_total_limit (`int`, *optional*):
        save_safetensors (`bool`, *optional*, defaults to `False`):
        save_on_each_node (`bool`, *optional*, defaults to `False`):
        no_cuda (`bool`, *optional*, defaults to `False`):
        seed (`int`, *optional*, defaults to 42):
        data_seed (`int`, *optional*):
        jit_mode_eval (`bool`, *optional*, defaults to `False`):
        use_ipex (`bool`, *optional*, defaults to `False`):
        bf16 (`bool`, *optional*, defaults to `False`):
        fp16 (`bool`, *optional*, defaults to `False`):
        fp16_opt_level (`str`, *optional*, defaults to 'O1'):
        fp16_backend (`str`, *optional*, defaults to `"auto"`):
        half_precision_backend (`str`, *optional*, defaults to `"auto"`):
        bf16_full_eval (`bool`, *optional*, defaults to `False`):
        fp16_full_eval (`bool`, *optional*, defaults to `False`):
        tf32 (`bool`, *optional*):
        local_rank (`int`, *optional*, defaults to -1):
        ddp_backend (`str`, *optional*):
        tpu_num_cores (`int`, *optional*):
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
        eval_steps (`int` or `float`, *optional*):
        dataloader_num_workers (`int`, *optional*, defaults to 0):
        past_index (`int`, *optional*, defaults to -1):
        run_name (`str`, *optional*):
        disable_tqdm (`bool`, *optional*):
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
        label_names (`List[str]`, *optional*):
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
        metric_for_best_model (`str`, *optional*):
        greater_is_better (`bool`, *optional*):
            Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
            should have a greater metric or not. Will default to:

            - `True` if `metric_for_best_model` is set to a value that isn't `"loss"` or `"eval_loss"`.
            - `False` if `metric_for_best_model` is not set, or set to `"loss"` or `"eval_loss"`.
        ignore_data_skip (`bool`, *optional*, defaults to `False`):
        sharded_ddp (`bool`, `str` or list of [`~trainer_utils.ShardedDDPOption`], *optional*, defaults to `False`):
            A list of options along the following:
            - `"simple"`: to use first instance of sharded DDP released by fairscale (`ShardedDDP`) similar to ZeRO-2.
            - `"zero_dp_2"`: to use the second instance of sharded DPP released by fairscale (`FullyShardedDDP`) in
              Zero-2 mode (with `reshard_after_forward=False`).
            - `"zero_dp_3"`: to use the second instance of sharded DPP released by fairscale (`FullyShardedDDP`) in
              Zero-3 mode (with `reshard_after_forward=True`).
            - `"offload"`: to add ZeRO-offload (only compatible with `"zero_dp_2"` and `"zero_dp_3"`).
        fsdp (`bool`, `str` or list of [`~trainer_utils.FSDPOption`], *optional*, defaults to `False`):
            - `"full_shard"`: Shard parameters, gradients and optimizer states.
            - `"shard_grad_op"`: Shard optimizer states and gradients.
            - `"offload"`: Offload parameters and gradients to CPUs (only compatible with `"full_shard"` and
              `"shard_grad_op"`).
            - `"auto_wrap"`: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.
        fsdp_config (`str` or `dict`, *optional*):
            A List of config and its options:
                - fsdp_min_num_params (`int`, *optional*, defaults to `0`):
                - fsdp_transformer_layer_cls_to_wrap (`List[str]`, *optional*):
                - fsdp_backward_prefetch (`str`, *optional*)
                    A list of options along the following:
                    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter's
                      gradient
                    - `"backward_post"` : This prefetches the next set of parameters after the current set of
                      parameter’s
                - fsdp_forward_prefetch (`bool`, *optional*, defaults to `False`)
                - limit_all_gathers (`bool`, *optional*, defaults to `False`)
                - xla (`bool`, *optional*, defaults to `False`):
                - xla_fsdp_settings (`dict`, *optional*)
                - xla_fsdp_grad_ckpt (`bool`, *optional*, defaults to `False`):
        deepspeed (`str` or `dict`, *optional*):
        label_smoothing_factor (`float`, *optional*, defaults to 0.0):
        debug (`str` or list of [`~debug_utils.DebugOption`], *optional*, defaults to `""`):
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
        optim_args (`str`, *optional*):
        group_by_length (`bool`, *optional*, defaults to `False`):
        length_column_name (`str`, *optional*, defaults to `"length"`):
        report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
        ddp_find_unused_parameters (`bool`, *optional*):
        ddp_bucket_cap_mb (`int`, *optional*):
        dataloader_pin_memory (`bool`, *optional*, defaults to `True`):
        skip_memory_metrics (`bool`, *optional*, defaults to `True`):
        push_to_hub (`bool`, *optional*, defaults to `False`):
        resume_from_checkpoint (`str`, *optional*):
        hub_model_id (`str`, *optional*):
        hub_strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
            Defines the scope of what is pushed to the Hub and when. Possible values are:
            - `"end"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and a
            - `"every_save"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and
            - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
            - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the output
        hub_token (`str`, *optional*):
        hub_private_repo (`bool`, *optional*, defaults to `False`):
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
        include_inputs_for_metrics (`bool`, *optional*, defaults to `False`):
        auto_find_batch_size (`bool`, *optional*, defaults to `False`)
        full_determinism (`bool`, *optional*, defaults to `False`)
        torchdynamo (`str`, *optional*):
        ray_scope (`str`, *optional*, defaults to `"last"`):
        ddp_timeout (`int`, *optional*, defaults to 1800):
        use_mps_device (`bool`, *optional*, defaults to `False`):
        torch_compile (`bool`, *optional*, defaults to `False`):
        torch_compile_backend (`str`, *optional*):
        torch_compile_mode (`str`, *optional*):
    """
```

### 训练细节

> [训练细节说明 · 3dot141/MedicalGPT Wiki · GitHub](https://github.com/3dot141/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82%E8%AF%B4%E6%98%8E)

### 踩坑

1. [Fix resuming PeftModel checkpoints in Trainer by llohann-speranca · Pull Request #24274 · huggingface/transformers · GitHub](https://github.com/huggingface/transformers/pull/24274) 使用 lora 尝试恢复的 checkpoints 的时候，发现恢复失败，参看 [训练细节](#训练细节) 发现依然存在问题。查询 `transformers` 的 issues 后，发现存在问题还没发布。所以魔改了一遍。
2. 尝试 merge lora 权重的时候，报错 `set_input_embeddings > NotImplementedError` 对比源码后，发现 [ChatGLM](https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py) 不存在问题，是因为实现了相关的源码。
	1. 见对比图
	2. ![505](docs/0652c68ed511cc1def7d1313868a8643_MD5.png)

## TODO

> 各个模型参数的含义详解。比如 fp16 int8 之类的如何调整。
