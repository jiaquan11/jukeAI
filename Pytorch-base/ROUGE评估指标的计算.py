from rouge import Rouge

#生成文本
generated_text = 'This is some generated text.'
#参考文本列表
reference_texts = ['This is a reference text.', 'This is another generated reference text.']

#计算ROUGE指标
rouge = Rouge()
scores = rouge.get_scores(generated_text, reference_texts[1])
print(f'scores: {scores}')

if __name__ == '__main__':
    #打印结果
    print("ROUGE-1 precision is:", scores[0]['rouge-1']['p'])
    print("ROUGE-1 recall is:", scores[0]['rouge-1']['r'])
    print("ROUGE-1 F1 score is:", scores[0]['rouge-1']['f'])