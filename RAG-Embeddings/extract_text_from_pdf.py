from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

#文档的加载与切割
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从PDF文件中(按指定页码)提取文字'''
    paragraphs = [] #存储提取的段落
    buffer = '' # 用于临时存储段落内容
    full_text = '' # 用于存储所有提取的文本
    #提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)): #遍历每一页
        #如果指定了页码范围，跳过范围外的页码
        if page_numbers is not None and i not in page_numbers:
            continue

        for element in page_layout: #element表示一个段落，每个段落中包含多个文本行
            if isinstance(element, LTTextContainer):#如果是文本容器
                # 提取文本  这里element就是一个文本段落，每一行的行尾都默认有换行符，full_text是一个文本段落的集合,每个文本段落之间先用换行符分隔做标记
                #将每个文本段落的文本提取出来，存储在full_text中，每个文本段落之间用换行符分隔
                full_text += element.get_text() + '\n' #获取到每个文本段落后，手动添加一个换行符，用于分隔每个文本段落
    #按空行分隔，将文本重新组织成段落
    # 按换行符分割文本，得到每一行的文本字符串，lines是一个文本行的集合。
    #因为前面对于段落的提取用换行符做了分隔，所以这里split是去掉了每一行的换行符，但是保留了段落之间的换行符
    #用于下面将文本重新组织成一个个的段落
    lines = full_text.split('\n')
    for text in lines:#遍历每一行文本，text是每一行的文本
        if len(text) > min_line_length:
            # 如果文本长度大于min_line_length，将文本添加到buffer中，如果文本不是以'-'结尾，添加一个空格
            # 如果文本是以'-'结尾，去掉'-',用于将一个段落的文本拼接在一起成一个长的文本字符串
            #这里没有连接符，默认都会在每一行的行首添加一个空格，这样就会导致段落的第一行的行首会有一个空格，这样是不准确的
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            # 如果是空行或者是长度小于min_line_length的行，将buffer中的文本添加到paragraphs中，并清空buffer，
            # 这里会丢弃空行或者长度小于min_line_length的行，应该会导致不准确，会丢失一些文本行(字数少的文本行会被直接丢弃)
            paragraphs.append(buffer)#将buffer中的文本添加到paragraphs中,buffer是一个拼接好的段落
            buffer = ''
    if buffer:#用于文章的最后一个段落的最后一行符合要求的文本(因为上面的循环中最后一个文本行不会被添加到paragraphs中)
        paragraphs.append(buffer)
    return paragraphs

if __name__ == '__main__':
    #测试
    paragraphs = extract_text_from_pdf('llama2.pdf', min_line_length=10)
    #paragraphs = extract_text_from_pdf("llama2.pdf", page_numbers=[2, 3])
    for para in paragraphs[:4]:
        print(para + '\n')