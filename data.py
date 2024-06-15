import pandas as pd
import os

if not os.path.exists('./dist'):
    os.makedirs('./dist')

ruozhiba = pd.read_json('./data/ruozhiba_qa.json')
alpaca = pd.read_json('./data/alpaca_data_cleaned.zh.json')
gpt4 = pd.read_json('./data/extracted_tagengo_gpt4.jsonl',lines=True)

# 定义函数，获取每个字符串中的字符并去重
def get_unique_chars(s):
    return ''.join(set(s))

ruozhiba_result = ruozhiba[ruozhiba['output'].str.len().between(20, 200)]
ruozhiba_result.loc[:, 'output'] =  ruozhiba_result['output'].apply(get_unique_chars)


alpaca_result = alpaca[alpaca['output'].str.len().between(200, 400)]
alpaca_result.loc[:, 'output'] =  alpaca_result['output'].apply(get_unique_chars)

gpt4_result_1k = gpt4[gpt4['output'].str.len().between(400, 1000)]
gpt4_result_1k.loc[:, 'output'] =  gpt4_result_1k['output'].apply(get_unique_chars)


gpt4_result_2k = gpt4[gpt4['output'].str.len().between(1000, 2000)]
gpt4_result_2k.loc[:, 'output'] =  gpt4_result_2k['output'].apply(get_unique_chars)


print(ruozhiba_result.shape)
print(alpaca_result.shape)
print(gpt4_result_1k.shape)
print(gpt4_result_2k.shape)

alpaca_result.to_csv('./dist/alpaca_data_cleaned.zh.csv',  columns=['output'],index=False,encoding='utf_8_sig')
ruozhiba_result.to_csv('./dist/ruozhiba_qa.csv', columns=['output'],index=False,encoding='utf_8_sig')
gpt4_result_1k.to_csv('./dist/gpt4_result_1k.csv', columns=['output'],index=False,encoding='utf_8_sig')
gpt4_result_2k.to_csv('./dist/gpt4_result_2k.csv', columns=['output'],index=False,encoding='utf_8_sig')