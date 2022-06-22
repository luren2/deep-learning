import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'}

for i in range(1,11):
    #因为一页只有25个电影简介，所以需要10页
    href='https://movie.douban.com/top250?start='
    res=requests.get(href,headers=headers)
    #头等信息
    html=res.text
    soup=BeautifulSoup(html,'html.parser') #解析网页代码
    #需要爬取 序号，电影名，评分，推荐语，链接
    movies=soup.find('ol',class_='grid_view').find_all('li')
    for movie in movies:
        movie_num=movie.find('div',class_='pic').find('em') #电影序号
        movie_name=movie.find('div',class_='info').find('span',class_='title') #电影名称
        movie_star=movie.find('div',class_='star').find('span',class_='rating_num') #电影评分
        movie_recommend=movie.find('p',class_='quote') #电影推荐语
        movie_href=movie.find('div',class_='pic').find('a')#电影链接 movie_href['href']
        if all([movie_name,movie_href,movie_num,movie_star,movie_recommend]):#表示都不为空
            print('排行第'+movie_num.text+'的电影:'+'\n')
            print('电影名：'+movie_name.text.strip()+'\t'+'评分：'+movie_star.text)
            print('推荐语：'+movie_recommend.text+'\n')
            print('链接：'+movie_href['href'])
            print('\n'+'******************************'+'\n')
        else:
            pass