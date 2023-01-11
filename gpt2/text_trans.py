with open('test_webnlg.log', 'r', encoding='utf-8') as f:
    with open('webnlg_test.txt','w',encoding='utf-8') as fw:
        for line in f:
            if line.find('===')!=-1:
                continue
            elif line.strip()=='':
                continue
            else:
                # print(line.strip())
                fw.write(line.strip())
                fw.write('\n')
            