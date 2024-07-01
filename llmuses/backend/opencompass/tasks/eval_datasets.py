# Copyright (c) Alibaba, Inc. and its affiliates.
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from opencompass.configs.datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from opencompass.configs.datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets
    from opencompass.configs.datasets.CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets
    from opencompass.configs.datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from opencompass.configs.datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    from opencompass.configs.datasets.CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets
    from opencompass.configs.datasets.CLUE_cmnli.CLUE_cmnli_gen_1abf97 import cmnli_datasets
    from opencompass.configs.datasets.CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets
    from opencompass.configs.datasets.FewCLUE_bustm.FewCLUE_bustm_gen_634f41 import bustm_datasets
    from opencompass.configs.datasets.FewCLUE_chid.FewCLUE_chid_gen_0a29a2 import chid_datasets
    from opencompass.configs.datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen_c68933 import cluewsc_datasets
    from opencompass.configs.datasets.FewCLUE_csl.FewCLUE_csl_gen_28b223 import csl_datasets
    from opencompass.configs.datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_gen_740ea0 import eprstmt_datasets
    from opencompass.configs.datasets.FewCLUE_ocnli_fc.FewCLUE_ocnli_fc_gen_f97a97 import ocnli_fc_datasets
    from opencompass.configs.datasets.FewCLUE_tnews.FewCLUE_tnews_gen_b90e4a import tnews_datasets
    from opencompass.configs.datasets.lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    from opencompass.configs.datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from opencompass.configs.datasets.storycloze.storycloze_gen_7f656a import storycloze_datasets
    from opencompass.configs.datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen_4dfefa import AX_b_datasets
    from opencompass.configs.datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_gen_68aac7 import AX_g_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    from opencompass.configs.datasets.SuperGLUE_CB.SuperGLUE_CB_gen_854c6c import CB_datasets
    from opencompass.configs.datasets.SuperGLUE_COPA.SuperGLUE_COPA_gen_91ca53 import COPA_datasets
    from opencompass.configs.datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_gen_27071f import MultiRC_datasets
    from opencompass.configs.datasets.SuperGLUE_RTE.SuperGLUE_RTE_gen_68aac7 import RTE_datasets
    from opencompass.configs.datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
    from opencompass.configs.datasets.race.race_gen_69ee4f import race_datasets
    from opencompass.configs.datasets.Xsum.Xsum_gen_31397e import Xsum_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.summedits.summedits_gen_315438 import summedits_datasets
    from opencompass.configs.datasets.math.math_gen_265cce import math_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
    from opencompass.configs.datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from opencompass.configs.datasets.siqa.siqa_gen_e78df3 import siqa_datasets
    from opencompass.configs.datasets.strategyqa.strategyqa_gen_1180a7 import strategyqa_datasets
    from opencompass.configs.datasets.winogrande.deprecated_winogrande_gen_a9ede5 import winogrande_datasets
    from opencompass.configs.datasets.obqa.obqa_gen_9069e4 import obqa_datasets
    from opencompass.configs.datasets.nq.nq_gen_c788f6 import nq_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets   # TODO: debug

    # Note: to be supported
    # from opencompass.configs.datasets.flores.flores_gen_806ede import flores_datasets
    # from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets
    # from opencompass.configs.datasets.commonsenseqa.commonsenseqa_gen_c946f2 import commonsenseqa_datasets


datasets = []
_locals = {k: v for k, v in locals().items() if k.endswith('_datasets')}

for k, v in _locals.items():
    for _dataset in v:
        _dataset['dataset_name'] = k.replace('_datasets', '')
        datasets.append(_dataset)


if __name__ == '__main__':
    for _dataset in datasets:
        print(_dataset)
