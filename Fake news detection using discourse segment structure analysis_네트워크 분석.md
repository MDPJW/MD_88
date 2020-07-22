 # Fake news detection using discourse segment structure analysis 
 
 * 가짜뉴스 탐지 사용 네트워크 분석
 
|저자|논문명|가짜뉴스 탐지에 사용된 네트워크 및 알고리즘|결과|
|------|---|---|----|
|Monu Waskale, Prof.Pritesh Jain, 2019|A Review Rumors Detection on Twitter Using Machine Learning Techniques|GRU-2 와 Tanh-RNN 분류기 사용|GRU-2분류기 12시간 이내의 트위터에 대해 83.9%의 정확성
|Cody Buntain, Jennifer Golbeck, 2018|Automatically Identifying Fake News in Popular Twitter Threads|제안하는 가짜뉴스 탐지 자동화 시스템|PHEME과 CREDBANK 데이터셋에 대해 66.93%와 70.28%의 정확성 
|Kursuncu et al. , 2018|Predictive Analysis on Twitter:Techniques and Applications|지도학습|기존결과:73%, 지도학습 후 결과:88.2~94.3%|
|Jiawei Zhang, Bowen Dong, Philip S. Yu, 2019|FAKEDETECTOR: Effective Fake News Detection|심층확산 네트워크|FAKEDETECTOR는 BI-Class Inference:14.5%,Multi-Class Inference:40% 더 좋음|
|Ruchansky et al. , 2017|CSI: A Hybrid Deep Model for Fake News Detection|제안하는 3가지 CSI모델|Weibo :0.867%, Twitter:0.631% 탐지 정확성|
|Monther Aldwairi, Ali Alwahedi|Detecting Fake News in Social Media Networks|Logistic 분류기|Clickbait 탐지율:99.4%|
|Traylor et al. ,2019|Classifying Fake News Articles Using Natural Language Processing to Identify In-Article Attribution as a Supervised LearningEstimator|ML 분류기|특징점이 하나인 가짜뉴스문서에 대한 탐지율:0.69% 분류오류:0.31%|
|Oshikawa et al. ,2018|A Survey on Natural Language Processing for Fake News Detection|GCN 모델|GCN 모델 최대 탐지율:94.4%|
|Gurav et al. ,2019|Series of methods accomplished by Machine Leaning|Pure NLP perspective towards false news detection|
|Agarwalla et al. ,2019|Accuracy in Naïve Bayes classifier with lid stone smoothing is 83% and in Naïve Bayes (without lidstone smoothing) is 74%.|An algorithm have been explored that can distinguish the difference between the fake and true news|
|Zellers et al. ,2019|Grover-Large yields 78% accuracy, 92% when dataset is increased.|Investigated the threats posed by adversaries seeking to spread disinformation and the possibilities of machine generated fake news.|
|Sivasangar i V et al. ,2018|Precision: 0.86 F1-score: 0.86|Rumor Detection by lever maturing the setting going before a tweet with a consecutive classifier|
|O'Brien et al. ,2018|Accuracy: 93.5% ± 0.2.|Deep neural networks to capture steady differences in the language of fake and real news: signatures of exaggeration and other forms of rhetoric| 
|Shu et al. ,2017|Studying existing literature in two segments: detection and characterization.|Datasets, evaluation metrics, and promising future ways in fake news detection discussed.|
|Silva et al. ,2019|ANN achieving 75% accuracy|An amalgamation of classic techniques with neural network|
|Dong et al. ,2019|Detect fake news from PHEME datasets using labeled data and unlabeled data.|Deep semi-supervised learning model by constructing two-path CNN|
|Yang et al. ,2019|88.063% accuracy|Soft labels are used to fine-tune NLI models, BERT, and the Decomposable Attention model. NLI models are trained independently and ensemble with a BERT model to define the soft labels.|
|Monti et al.,2019|More than 93% accuracy achieved by automatic fake news detection model built on geometric deeplearning.|The underlying core algorithms allow for a fusion of dissimilar data such as content, profile, activity of a user, social graph and propagation pattern of news, which is achieved by generalizing CNN to graphs.|
|Ajao et al.,2018|82% accuracy via both text and images by automatic identification of features within Twitter posts|A hybrid deep learning model of LSTM and CNN models is used.|
|Thota et al. ,2018|94.21% accuracy. The Dense Neural Network beats existing model architectures by 2.5%.|A finely tuned TF-IDF Dense neural network architecture to predict the stance between a given pair of headline and article body.|
|Helmstetter, Heiko Paulheim ,2018|F1 score of 0.9 achieved|Trustworthy or untrustworthy source are used to automaticallylabel the data during collection, and train a classifier on this dataset.|
|Atodiresei et al. ,2018|Tweet score can be 1000, -500 or in [-50,100] User score can be in [0,12]|Credibility score fromhashtag sentiments, emoji sentiments, text sentiments and namedentity recognition. Higher the credibility score, higher the trust|
|Hamid Karimi, Jiliang Tang ,2019|82% accuracy|Hierarchical Discourse level structural data analysis for fake newsdetection. A structure is trained on the dataset, quantifiable properties of which are used for classification process in the model.|
|Shuo Yang et al., 2019|Graphical model is built taking into account reliability of the news and credibility score of the user. 75.9% max. Accuracy accomplished on LIAR dataset.|Unsupervised method is investigated. Opinion is extracted from hierarchy social engagement information acquired from social media users..Reality and credibility is considered by an efficient Gibbs-sampling method.|
|Zhou et al.,2018|Models so far possess a greater possibility to misclassify fake news that tampers with facts as well as under-written real news articles|Simply looking into Linguistic aspects is not enough for fake news detection.|
|Álvaro and Lara,2019|93% accuracy with superior metrics compared to other deep learning models|BERT, LSTM and Convolutional Neural Network models are trained based merely on textual features.|
