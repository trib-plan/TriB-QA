# -*- coding：utf-8 -*-

# from data.train_preprocessed.evaluation_metric.rouge import RougeL
from rouge import RougeL
'''
依赖的这个脚本在 train_preprocessed.evaluation_metric 中
'''

def normalize(s):
    """
    Normalize strings to space joined chars.
    Args:
        s: a list of strings.
    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        norm_s = ''.join(tokens)
        norm_s = norm_s.replace(u"，", u",")
        norm_s = norm_s.replace(u"。", u".")
        norm_s = norm_s.replace(u"！", u"!")
        norm_s = norm_s.replace(u"？", u"?")
        norm_s = norm_s.replace(u"；", u";")
        norm_s = norm_s.replace(u"（", u"(").replace(u"）", u")")
        norm_s = norm_s.replace(u"【", u"[").replace(u"】", u"]")
        norm_s = norm_s.replace(u"“", u"\"").replace(u"“", u"\"")
        normalized.append(norm_s)
    return normalized

def rougeL(pred_text_list, ref_text_list, yn_label=None, yn_ref=None):
    pred_text_list = normalize(pred_text_list)
    ref_text_list = normalize(ref_text_list)
    pred_text = pred_text_list[0]
    rouge_eval = RougeL(alpha=1.0, beta=1.0, gamma=1.2)
    rouge_eval.add_inst(pred_text, ref_text_list, yn_label=yn_label, yn_ref=yn_ref)
    return rouge_eval.score()

def get_rougel(text, ref_list):
    '''
    text: string
    ref_list: List include ref_texts
    '''
    return rougeL([text], ref_list)

if __name__ == '__main__':
    text = '今天天气'
    ref_list = ['今天天气好',]
    print(main(text, ref_list))
