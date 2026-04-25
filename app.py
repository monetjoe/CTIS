import os
import torch
import shutil
import librosa
import warnings
import numpy as np
import gradio as gr
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
from model import EvalNet
from utils import get_modelist, find_files, embed_img, _L, EN_US


TRANSLATE = {
    "C0090": ["大笒", "da4_cen2"],
    "C0091": ["高音横笛", "Treble_heng2_di2"],
    "C0092": ["低音横笛", "Bass_heng2_di2"],
    "C0093": ["中音横笛", "Alto_heng2_di2"],
    "C0094": ["唢呐", "suo3_na"],
    "C0095": ["长唢呐", "chang2_suo3_na"],
    "C0096": ["小筚篥", "Treble_bi4_li"],
    "C0097": ["中音筚篥", "Alto_bi4_li"],
    "C0098": ["低音筚篥", "Bass_bi4_li"],
    "C0099": ["短箫", "duan3_xiao1"],
    "C0100": ["短箫(传统)", "duan3_xiao1_(traditional)"],
    "C0101": ["洞箫", "dong4_xiao1"],
    "C0113": ["尖子号", "jian1_zi3_hao4"],
    "C0114": ["尖子号2", "jian1_zi3_hao4_2"],
    "C0117": ["南音洞箫", "nan2_yin1_dong4_xiao1"],
    "C0123": ["南嗳仔", "nan2_ai1_zai3"],
    "C0124": ["大吹", "da4_chui1"],
    "C0182": ["长号", "chang2_hao4"],
    "C0183": ["老长号", "lao3_chang2_hao4"],
    "C0187": ["高音唢呐", "Treble_suo3_na"],
    "C0188": ["低音唢呐", "Bass_suo3_na"],
    "C0200": ["大芦笙", "da4_lu2_sheng1"],
    "C0201": ["小芦笙", "xiao3_lu2_sheng1"],
    "C0237": ["G调梆笛", "bang1_di2_in_G"],
    "C0243": ["高音键笙", "Treble_jian4_sheng1"],
    "C0244": ["传统笙", "traditional_sheng1"],
    "C0257": ["低音加键唢呐", "Bass_jia1_jian4_suo3_na"],
    "C0259": ["中音加键唢呐", "Alto_jia1_jian4_suo3_na"],
    "C0263": ["中音笙", "Alto_sheng1"],
    "C0264": ["低音笙", "Bass_sheng1"],
    "C0265": ["管子", "guan3_zi"],
    "C0280": ["A调曲笛", "qu3_di2_in_A"],
    "C0281": ["G调新笛", "xin1_di2_in_G"],
    "C0282": ["萧", "xiao1"],
    "C0283": ["埙", "xun1"],
    "C0296": ["唢呐2", "suo3_na_2"],
    "C0303": ["小闷笛", "xiao3_men1_di2"],
    "C0304": ["侗笛", "dong4_di2"],
    "C0305": ["德", "de2"],
    "C0306": ["拉祜族葫芦笙", "la1_hu2_zu2_hu2_lu2_sheng1"],
    "C0308": ["吐良", "tu3_liang2"],
    "C0309": ["葫芦丝", "hu2_lu2_si1"],
    "C0310": ["F调巴乌", "ba1_wu1_in_F"],
    "C0311": ["俄比", "e2_bi3"],
    "C0316": ["侗巴", "dong4_ba1"],
    "D0015": ["扬琴", "yang2_qin2"],
    "D0048": ["低音大锣", "Bass_da4_luo2"],
    "D0049": ["虎音锣", "hu3_yin1_luo2"],
    "D0050": ["小钹", "xiao3_bo1"],
    "D0051": ["钹", "bo1"],
    "D0058": ["提手(板)", "ti2_shou3_(ban3)"],
    "D0060": ["川小锣", "chuan1_xiao3_luo2"],
    "D0061": ["大铛铛", "da4_cheng1_cheng1"],
    "D0062": ["小铛铛", "xiao3_cheng1_cheng1"],
    "D0063": ["二馨", "er4_xin1"],
    "D0064": ["川大钵", "chuan1_da4_bo1"],
    "D0065": ["苏钵", "su1_bo1"],
    "D0066": ["川剧堂鼓", "chuan1_ju4_tang2_gu3"],
    "D0067": ["川铰", "chuan1_jiao3"],
    "D0068": ["川大锣", "chuan1_da4_luo2"],
    "D0069": ["蛮锣", "man2_luo2"],
    "D0070": ["包锣", "bao1_luo2"],
    "D0071": ["引鼓", "yin3_gu3"],
    "D0102": ["上杖鼓", "shang4_zhang4_gu3"],
    "D0103": ["小锣", "xiao3_luo2"],
    "D0104": ["圆锣", "yuan2_luo2"],
    "D0105": ["杖鼓", "zhang4_gu3"],
    "D0125": ["南鼓", "nan2_gu3"],
    "D0126": ["压脚鼓", "ya1_jiao3_gu3"],
    "D0127": ["钟", "zhong1"],
    "D0128": ["草锣", "cao3_luo2"],
    "D0129": ["锣仔", "luo2_zai3"],
    "D0130": ["响盏", "xiang3_zhan3"],
    "D0131": ["小叫", "xiao3_jiao4"],
    "D0132": ["拍", "pai1"],
    "D0137": ["渔鼓", "yu2_gu3"],
    "D0138": ["简板", "jian3_ban3"],
    "D0140": ["脚梆子", "jiao3_bang1_zi"],
    "D0143": ["双铃", "shuang1_ling2"],
    "D0144": ["小叫锣", "xiao3_jiao4_luo2"],
    "D0145": ["拍板", "pai1_ban3"],
    "D0146": ["四宝", "si4_bao3"],
    "D0147": ["响盏2", "xiang3_zhan3_2"],
    "D0172": ["碗碗", "wan3_wan3"],
    "D0173": ["代子", "dai4_zi"],
    "D0176": ["福(新)", "fu2_(reformed)"],
    "D0177": ["禄(新)", "lu4_(reformed)"],
    "D0178": ["寿(新)", "shou4_(reformed)"],
    "D0179": ["宜春三星鼓福鼓老鼓", "yi2_chun1_san1_xing1_gu3_fu2_gu3_(traditional)"],
    "D0180": ["宜春三星鼓禄鼓老鼓", "yi2_chun1_san1_xing1_gu3_lu4_gu3_(traditional)"],
    "D0181": ["宜春三星鼓寿鼓老鼓", "yi2_chun1_san1_xing1_gu3_shou4_gu3_(traditional)"],
    "D0184": ["宜春三星鼓双铛", "yi2_chun1_san1_xing1_gu3_shuang1_ding1"],
    "D0185": ["宜春三星鼓单铛", "yi2_chun1_san1_xing1_gu3_dan1_ding1"],
    "D0186": ["宜春三星鼓镲", "yi2_chun1_san1_xing1_gu3_chao3"],
    "D0241": ["编钟", "bian1_zhong1"],
    "D0242": ["编磬", "bian1_qing4"],
    "D0245": ["南梆子", "nan2_bang1_zi"],
    "D0246": ["北梆子", "bei3_bang1_zi"],
    "D0247": ["碰铃", "peng4_ling2"],
    "D0248": ["中国大鼓", "Chinese_da4_gu3"],
    "D0249": ["花盆鼓", "hua1_pen2_gu3"],
    "D0250": ["小堂鼓", "xiao3_tang2_gu3"],
    "D0251": ["扁鼓", "bian3_gu3"],
    "D0252": ["五音排鼓", "wu3_yin1_pai2_gu3"],
    "D0268": ["草帽镲", "cao3_mao4_chao3"],
    "D0269": ["铙", "nao2"],
    "D0270": ["铙钹", "nao2_bo1"],
    "D0271": ["小镲", "xiao3_chao3"],
    "D0272": ["抄锣", "chao1_luo2"],
    "D0273": ["中虎", "zhong1_hu3"],
    "D0274": ["武锣", "wu3_luo2"],
    "D0275": ["小锣2", "xiao3_luo2_2"],
    "D0276": ["马锣", "ma3_luo2"],
    "D0277": ["木鱼", "mu4_yu2"],
    "D0278": ["板鼓", "ban3_gu3"],
    "D0279": ["云锣", "yun2_luo2"],
    "D0284": ["斗锣", "dou3_luo2"],
    "D0286": ["曲锣", "qu3_luo2"],
    "D0287": ["深波", "shen1_bo1"],
    "D0290": ["大镲", "da4_chao3"],
    "D0298": ["编铓", "bian1_zhang1"],
    "D0299": ["牛铃", "niu2_ling2"],
    "D0315": ["竹排琴", "zhu2_pai2_qin2"],
    "D0325": ["那格拉", "na4_ge2_la1"],
    "D0326": ["库休克", "ku4_xiu1_ke4"],
    "D0327": ["萨巴依", "sa4_ba1_yi1"],
    "D0328": ["手鼓", "shou3_gu3"],
    "L0044": ["锡剧主胡", "xi1_ju4_zhu3_hu2"],
    "L0045": ["扬剧主胡", "yang2_ju4_zhu3_hu2"],
    "L0046": ["扬剧主胡F调", "yang2_ju4_zhu3_hu2_in_F"],
    "L0047": ["扬剧主胡(小西皮)", "yang2_ju4_zhu3_hu2_(xiao3_xi1_pi2)"],
    "L0053": ["广西彩调主胡", "guang3_xi1_cai3_diao4_zhu3_hu2"],
    "L0055": ["牛腿琴", "niu2_tui3_qin2"],
    "L0056": ["壮剧马骨胡D调", "zhuang4_ju4_ma3_gu3_hu2_in_D"],
    "L0072": ["盖板(新)D调", "gai4_ban3_(reformed)_in_D"],
    "L0073": ["盖板(传统)", "gai4_ban3_(traditional)"],
    "L0074": ["壮剧土胡", "zhuang4_ju4_tu3_hu2"],
    "L0075": ["晋剧晋胡", "jin4_ju4_jin4_hu2"],
    "L0076": ["壮剧土胡2", "zhuang4_ju4_tu3_hu2_2"],
    "L0077": ["晋剧二股弦", "jin4_ju4_er4_gu3_xian2"],
    "L0080": ["吕剧坠琴", "lv3_ju4_zhui4_qin2"],
    "L0084": ["奚琴(传统)", "xi1_qin2_(traditional)"],
    "L0085": ["奚琴(改良)", "xi1_qin2_(reformed)"],
    "L0086": ["中音奚琴(改良)", "Alto_xi1_qin2_(reformed)"],
    "L0115": ["莱芜梆子-梆胡", "lai2_wu2_bang1_zi-bang1_hu2"],
    "L0121": ["六角弦", "liu4_jiao3_xian2"],
    "L0122": ["壳仔弦", "ke2_zai3_xian2"],
    "L0133": ["陇剧陇胡(传统)", "long3_ju4_long3_hu2_(traditional)"],
    "L0134": ["陇剧陇胡(改良)D调", "long3_ju4_long3_hu2_(reformed)_in_D"],
    "L0135": ["齐琴", "qi2_qin2"],
    "L0136": ["渔胡", "yu2_hu2"],
    "L0139": ["坠胡", "zhui4_hu2"],
    "L0141": ["越胡", "yue4_hu2"],
    "L0148": ["板胡", "ban3_hu2"],
    "L0149": ["绍剧板胡", "shao4_ju4_ban3_hu2"],
    "L0150": ["宛梆子梆胡", "yuan1_bang1_zi_bang1_hu2"],
    "L0151": ["四弦", "si4_xian2"],
    "L0152": ["滇葫(小二胡)", "dian1_hu2_(xiao3_er4_hu2)"],
    "L0153": ["云南花灯丝弦", "yun2_nan2_hua1_deng1_si1_xian2"],
    "L0154": ["仕胡", "shi4_hu2"],
    "L0155": ["伬胡", "chi4_hu2"],
    "L0156": ["工胡", "gong1_hu2"],
    "L0157": ["大胡", "da4_hu2"],
    "L0158": ["低音伬胡", "Bass_chi4_hu2"],
    "L0160": ["丝弦", "si1_xian2"],
    "L0161": ["滇胡", "dian1_hu2"],
    "L0162": ["襄阳专用胡琴", "xiang1_yang2_zhuan1_yong4_hu2_qin2"],
    "L0163": ["雷胡", "lei2_hu2"],
    "L0164": ["赣胡", "gan4_hu2"],
    "L0165": ["高腔赣胡", "gao1_qiang1_gan4_hu2"],
    "L0166": ["高腔赣胡第2代", "gao1_qiang1_gan4_hu2_2nd_generation"],
    "L0167": ["黔胡", "qian2_hu2"],
    "L0168": ["花胡", "hua1_hu2"],
    "L0169": ["花胡2", "hua1_hu2_2"],
    "L0170": ["二股弦", "er4_gu3_xian2"],
    "L0239": ["高音板胡", "Treble_ban3_hu2"],
    "L0240": ["中音板胡", "Alto_ban3_hu2"],
    "L0256": ["雷琴", "lei2_qin2"],
    "L0266": ["二胡", "er4_hu2"],
    "L0285": ["二弦", "er4_xian2"],
    "L0288": ["椰胡", "ye1_hu2"],
    "L0291": ["扁八角高胡", "bian3_ba1_jiao3_gao1_hu2"],
    "L0292": ["六角高胡", "liu4_jiao3_gao1_hu2"],
    "L0297": ["中胡", "zhong1_hu2"],
    "L0307": ["芦笙", "lu2_sheng1"],
    "L0312": ["牛角胡", "niu2_jiao3_hu2"],
    "L0313": ["佤族独弦琴", "wa3_zu2_du2_xian2_qin2"],
    "L0314": ["葫芦琴", "hu2_lu2_qin2"],
    "T0006": ["陶布舒尔", "tao2_bu4_shu1_er3"],
    "T0007": ["雅托嘎", "ya3_tuo1_ga2"],
    "T0078": ["四股弦", "si4_gu3_xian2"],
    "T0081": ["玄琴", "xuan2_qin2"],
    "T0082": ["伽倻琴(改良)", "jia1_ye2_qin2_(reformed)"],
    "T0083": ["伽倻琴", "jia1_ye2_qin2"],
    "T0087": ["雅筝", "ya3_zheng1"],
    "T0088": ["扬琴2", "yang2_qin2_2"],
    "T0089": ["扬琴3", "yang2_qin2_3"],
    "T0111": ["三弦", "san1_xian2"],
    "T0116": ["八角月琴", "ba1_jiao3_yue4_qin2"],
    "T0159": ["双清", "shuang1_qing1"],
    "T0171": ["月琴", "yue4_qin2"],
    "T0238": ["大阮", "da4_ruan3"],
    "T0254": ["箜篌", "kong1_hou2"],
    "T0255": ["古筝", "gu3_zheng1"],
    "T0260": ["中阮", "zhong1_ruan3"],
    "T0261": ["柳琴", "liu3_qin2"],
    "T0262": ["琵琶", "pi2_pa2"],
    "T0267": ["扬琴4", "yang2_qin2_4"],
    "T0289": ["三弦2", "san1_xian2_2"],
    "T0294": ["南音琵琶", "nan2_yin1_pi2_pa2"],
    "T0295": ["南音三弦", "nan2_yin1_san1_xian2"],
    "T0300": ["澜沧小三弦", "lan2_cang1_xiao3_san1_xian2"],
    "T0301": ["玎", "ding1"],
    "T0302": ["傈傈族奇奔", "li4_li4_zu2_qi2_ben1"],
    "T0317": ["独弦琴", "du2_xian2_qin2"],
    "T0318": ["弹拨尔", "dan4_bo1_er3"],
    "T0319": ["低音热瓦普", "Bass_re4_wa3_pu3"],
    "T0320": ["民间热瓦普", "folk_re4_wa3_pu3"],
    "T0323": ["都它尔", "du1_ta1_er3"],
}
CLASSES = list(TRANSLATE.keys())
TEMP_DIR = "./__pycache__/tmp"
SAMPLE_RATE = 44100


def circular_padding(spec: np.ndarray, end: int):
    size = len(spec)
    if end <= size:
        return spec

    num_padding = end - size
    num_repeat = num_padding // size + int(num_padding % size != 0)
    padding = np.tile(spec, num_repeat)
    return np.concatenate((spec, padding))[:end]


def wav2mel(audio_path: str, width=2, top_db=40):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    non_silents = librosa.effects.split(y, top_db=top_db)
    y = np.concatenate([y[start:end] for start, end in non_silents])
    total_frames = len(y)
    if total_frames % (width * sr) != 0:
        count = total_frames // (width * sr) + 1
        y = circular_padding(y, count * width * sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    total_frames = log_mel_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_mel_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{TEMP_DIR}/{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def wav2cqt(audio_path: str, width=2, top_db=40):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    non_silents = librosa.effects.split(y, top_db=top_db)
    y = np.concatenate([y[start:end] for start, end in non_silents])
    total_frames = len(y)
    if total_frames % (width * sr) != 0:
        count = total_frames // (width * sr) + 1
        y = circular_padding(y, count * width * sr)

    cqt_spec = librosa.cqt(y=y, sr=sr)
    log_cqt_spec = librosa.power_to_db(np.abs(cqt_spec) ** 2, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    total_frames = log_cqt_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_cqt_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{TEMP_DIR}/{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def wav2chroma(audio_path: str, width=2, top_db=40):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    non_silents = librosa.effects.split(y, top_db=top_db)
    y = np.concatenate([y[start:end] for start, end in non_silents])
    total_frames = len(y)
    if total_frames % (width * sr) != 0:
        count = total_frames // (width * sr) + 1
        y = circular_padding(y, count * width * sr)

    chroma_spec = librosa.feature.chroma_stft(y=y, sr=sr)
    log_chroma_spec = librosa.power_to_db(np.abs(chroma_spec) ** 2, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    total_frames = log_chroma_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_chroma_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{TEMP_DIR}/{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def most_frequent_value(lst: list):
    counter = Counter(lst)
    max_count = max(counter.values())
    for element, count in counter.items():
        if count == max_count:
            return element

    return None


def infer(wav_path: str, log_name: str, folder_path=TEMP_DIR):
    status = "Success"
    filename = result = None
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        if not wav_path:
            return None, "请输入音频!"

        spec = log_name.split("_")[-3]
        os.makedirs(folder_path, exist_ok=True)
        model = EvalNet(log_name, len(TRANSLATE)).model
        eval("wav2%s" % spec)(wav_path)
        jpgs = find_files(folder_path, ".jpg")
        preds = []
        for jpg in jpgs:
            input = embed_img(jpg)
            output: torch.Tensor = model(input)
            preds.append(torch.max(output.data, 1)[1])

        pred_id = most_frequent_value(preds)
        filename = os.path.basename(wav_path)
        result = (
            TRANSLATE[CLASSES[pred_id]][1].capitalize()
            if EN_US
            else f"{TRANSLATE[CLASSES[pred_id]][0]} ({TRANSLATE[CLASSES[pred_id]][1].capitalize()})"
        )

    except Exception as e:
        status = f"{e}"

    return status, filename, result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    models = get_modelist(assign_model="regnet_y_32gf_cqt")
    examples = []
    example_wavs = find_files()
    for wav in example_wavs:
        examples.append([wav, models[0]])

    with gr.Blocks() as demo:
        gr.Interface(
            fn=infer,
            inputs=[
                gr.Audio(label=_L("上传录音"), type="filepath"),
                gr.Dropdown(choices=models, label=_L("选择模型"), value=models[0]),
            ],
            outputs=[
                gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                gr.Textbox(label=_L("音频文件名"), buttons=["copy"]),
                gr.Textbox(label=_L("中国乐器识别"), buttons=["copy"]),
            ],
            examples=examples,
            cache_examples=False,
            flagging_mode="never",
            title=_L("建议录音时长保持在 3s 左右"),
        )

        gr.Markdown(
            f"# {_L('引用')}"
            + """
            ```bibtex
            @article{Zhou-2025,
                author  = {Monan Zhou and Shenyang Xu and Zhaorui Liu and Zhaowen Wang and Feng Yu and Wei Li and Baoqiang Han},
                title   = {CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research},
                journal = {Transactions of the International Society for Music Information Retrieval},
                volume  = {8},
                number  = {1},
                pages   = {22--38},
                month   = {Mar},
                year    = {2025},
                url     = {https://doi.org/10.5334/tismir.194},
                doi     = {10.5334/tismir.194}
            }
            ```"""
        )

    demo.launch(css="#gradio-share-link-button-0 { display: none; }")
