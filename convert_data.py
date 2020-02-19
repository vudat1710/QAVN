import pandas as pd
import json
from nltk import word_tokenize, sent_tokenize

class ConvertData:
    def get_train_random(self, filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                _list = line.split("\t")
                _list[2] = _list[2].strip().split(' ')
                _list[1] = _list[1].strip().split(' ')
                data.append(_list)
        f.close()
        return data

    def get_test_dev(self, filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                _list = line.split("\t")
                _list = _list[:3]
                _list[2] = _list[2].strip().split(' ')
                _list[1] = _list[1].strip().split(' ')
                neg_list = []
                for id in _list[2]:
                    if id not in _list[1]:
                        neg_list.append(id)
                del _list[2]
                _list.append(neg_list)
                data.append(_list)
        f.close()
        return data
    
    def get_text_body_by_qid(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                _list = line.split("\t")
                data[_list[0]] = _list[2].strip()
        return data
    
    def get_negative_sr(self, df):
        return df.iloc[:, 2]

    def get_random_negative(self, df, start, end, neg):
        negative_qs = df.iloc[:,2]
        similar_qs = df.iloc[:,1]
        a = similar_qs.apply(lambda x: len(x) in range(start, end + 1))
        similar_5_neg = similar_qs[a == True]
        negative_5_neg = negative_qs[a == True]
        negative_5_neg = negative_5_neg.apply(lambda x: x[:neg])
        qid_5_neg = df.iloc[:,0][a == True]
        qid_5_neg = pd.DataFrame({'qid': qid_5_neg})
        similar_5_neg = pd.DataFrame({'similar': similar_5_neg})
        negative_5_neg = pd.DataFrame({'negative': negative_5_neg})
        df_5_neg = pd.concat([qid_5_neg,similar_5_neg,negative_5_neg], axis=1)
        df_5_neg.to_json('askubuntu/data_%d_neg.json' % neg, orient='records')

    def tokenize_each_cell(self, paragraph):
        sent_list = sent_tokenize(paragraph)
        word_list = []
        for sent in sent_list:
            _list = word_tokenize(sent)
            word_list = word_list + _list
        return len(word_list)
    
    def get_mean_each_col(self, df):
        df['q1'] = df['q1'].apply(lambda x: self.tokenize_each_cell(x))
        df['q2'] = df['q2'].apply(lambda x: self.tokenize_each_cell(x))
        return (df['q1'].mean(), df['q2'].mean())
    
    def main(self):
        data = self.get_train_random('askubuntu/train_random.txt')
        df = pd.DataFrame(data)
        # self.get_random_negative(df, 1, 5, 5)
        # self.get_random_negative(df, 6, 15, 10)
        # self.get_random_negative(df, 16, 90, 40)
        # self.get_random_negative(df, 91, 623, 100)
        self.get_random_negative(df, 1, 623, 5)
        with open('askubuntu/data_5_neg.json', 'r') as f:
            data = json.load(f)
        f.close()
        # data = self.get_test_dev('askubuntu/dev.txt')
        # df = pd.DataFrame(data)
        df = pd.DataFrame(data, columns=['qid','similar', 'negative'])
        qid = df['qid']
        # print(df['similar'])
        with open('askubuntu/text_tokenized.txt', 'r') as f:
            data2 = []
            for line in f.readlines():
                _list = line.split("\t")
                _list[2] = _list[2].strip()
                _list[1] = _list[1].strip()
                data2.append(_list)
        f.close()
        df_text = pd.DataFrame(data2)
        count = 0
        with open('train.txt', 'w') as f:
            for id in qid:
                df_with_id = df.loc[df['qid'] == id]
                df_text_with_qid = df_text.loc[df_text.iloc[:,0] == id]
                if list(df_with_id['similar'])[0] != ['']:
                    for sim_id in list(df_with_id['similar'])[0]:
                        f.write(list(df_text_with_qid.iloc[:,1])[0])
                        f.write('\t')
                        # f.write(list(df_text_with_qid.iloc[:,2])[0])
                        # f.write('\t')
                        df_text_with_sim = df_text.loc[df_text.iloc[:,0] == sim_id]
                        f.write(list(df_text_with_sim.iloc[:,1])[0])
                        f.write('\t')
                        # f.write(list(df_text_with_qid.iloc[:,2])[0])
                        # f.write('\t')
                        f.write('1\n')
                for neg_id in list(df_with_id['negative'])[0]:
                    f.write(list(df_text_with_qid.iloc[:,1])[0])
                    f.write('\t')
                    # f.write(list(df_text_with_qid.iloc[:,2])[0])
                    # f.write('\t')
                    df_text_with_neg = df_text.loc[df_text.iloc[:,0] == neg_id]
                    f.write(list(df_text_with_neg.iloc[:,1])[0])
                    f.write('\t')
                    # f.write(list(df_text_with_qid.iloc[:,2])[0])
                    # f.write('\t')
                    f.write('0\n')
                count += 1
                print(count)
        f.close()


        # with open('data.txt', 'r') as f:
        #     data = []
        #     for line in f.readlines():
        #         _list = line.split('\t')
        #         _list = [x.strip() for x in _list]
        #         data.append(_list)
        # f.close()
        # df = pd.DataFrame(data, columns=['q1', 'q2', 'label'])
        # mean_1, mean_2 = self.get_mean_each_col(df)
        # print('%d, %d' % (mean_1, mean_2))



if __name__=="__main__":
    a = ConvertData()
    a.main()
        

