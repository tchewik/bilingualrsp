local batch_size = std.extVar("batch_size");
local combine_batches = std.extVar("combine_batches");
local corpus = std.extVar("corpus");
local cross_validation = std.extVar("cross_validation");
local cuda_device = std.parseInt(std.extVar("cuda_device"));
local data_manager_path = std.extVar("data_manager_path");
local discriminator_warmup = std.extVar("discriminator_warmup");
local dwa_bs = std.extVar("dwa_bs");
local edu_encoding_kind = std.extVar("edu_encoding_kind");
local emb_size = std.extVar("emb_size");
local foldnum = std.extVar("foldnum");
local freeze_first_n = std.extVar("freeze_first_n");
local grad_clipping_value = std.extVar("grad_clipping_value");
local hidden_size = std.extVar("hidden_size");
local if_edu_start_loss = std.extVar("if_edu_start_loss");
local lang = std.extVar("lang");
local lr = std.extVar("lr");
local lstm_bidirectional = std.extVar("lstm_bidirectional");
local rel_classification_kind = std.extVar("rel_classification_kind");
local run_name = std.extVar("run_name");
local save_path = std.extVar("save_path");
local second_lang_fold = std.extVar("second_lang_fold");
local second_lang_fraction = std.extVar("second_lang_fraction");
local seed = std.extVar("seed");
local segmenter_dropout = std.extVar("segmenter_dropout");
local segmenter_hidden_dim = std.extVar("segmenter_hidden_dim");
local segmenter_type = std.extVar("segmenter_type");
local token_bilstm_hidden = std.extVar("token_bilstm_hidden");
local transformer_name = std.extVar("transformer_name");
local use_crf = std.extVar("use_crf");
local use_discriminator = std.extVar("use_discriminator");
local use_log_crf = std.extVar("use_log_crf");
local window_size = std.extVar("window_size");
local window_padding = std.extVar("window_padding");

{
    "data": {
            "corpus": corpus,
            "lang": lang,
            "cross_validation": cross_validation,
            "fold": foldnum,
            "data_manager_path": data_manager_path,
            "second_lang_fraction": second_lang_fraction,
            "second_lang_fold": second_lang_fold,
    },
    "model": {
        "transformer": {
            "model_name": transformer_name,
            "emb_size": emb_size,
            "normalize": true,
            "freeze_first_n": freeze_first_n,
            "window_size": window_size,
            "window_padding": window_padding
        },
        "segmenter": {
            "type": segmenter_type,
            "use_crf": use_crf,
            "use_log_crf": use_log_crf,
            "hidden_dim": segmenter_hidden_dim,
            "lstm_dropout": 0.2,
            "lstm_num_layers": 1,
            "if_edu_start_loss": if_edu_start_loss,
            "lstm_bidirectional": lstm_bidirectional,
        },
        "hidden_size": hidden_size,
        "edu_encoding_kind": edu_encoding_kind,
        "rel_classification_kind": rel_classification_kind,
        "token_bilstm_hidden": token_bilstm_hidden,
        "use_discriminator": use_discriminator,
        "discriminator_warmup": discriminator_warmup,
        "dwa_bs": dwa_bs,
    },
    "trainer": {
            "lr": lr,
            "seed": seed,
            "epochs": 100,
            "use_amp": false,
            "lr_decay": 0.95,
            "patience": 7,
            "project": "rurst",
            "run_name": run_name,
            "eval_size": 30,
            "save_path": save_path,
            "batch_size": batch_size,
            "combine_batches": combine_batches,
            "weight_decay": 0.01,
            "lr_decay_epoch": 1,
            "lm_lr_mutliplier": 0.2,
            "grad_norm_value": 1.0,
            "grad_clipping_value": grad_clipping_value,
            "gpu": cuda_device,
    }
}