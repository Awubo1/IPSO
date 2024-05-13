import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
from matplotlib import rcParams
# mpl.rcParams['font.sans-serif'] = ['SimSun'] # 指定默认字体：解决plot不能显示中文问题
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

config = {
            "font.family": 'serif',
            "font.size": 15,
            "font.weight":"bold",
            "mathtext.fontset": 'stix',
            "font.serif": ['SimHei'],#宋体
            'axes.unicode_minus': False # 处理负号
         }
rcParams.update(config)
# 迭代130次
x = np.arange(0, 100)

# 六种算法的适应度模数据


fitness_pso = [1.0734263623009148, 0.9728571932500736, 0.8762071589269048, 0.8762071589269048, 0.8033957437229275, 0.7593816518800431, 0.7593816518800431, 0.7593816518800431, 0.7359889510686035, 0.7359889510686035, 0.7359889510686035, 0.7359889510686035, 0.7059889510686035, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552, 0.7005868044407552]
fitness_ga = [1.1314899588652622, 1.0863837260492393, 1.0813468614847639, 1.0803921027875992, 0.9751599510256488, 0.966058842187016, 0.9525654845303651, 0.8885723594199141, 0.8735120050699032, 0.8594198269299322, 0.8519985564755657, 0.84909565224070997, 0.838665811855105, 0.831857042511081, 0.826876287144957, 0.8228765839587175, 0.8165106570190993, 0.8114563886668068, 0.8074563886668068, 0.7954563886668068, 0.7924563886668068, 0.7914563886668068, 0.7884563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068,0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068,0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068, 0.7834563886668068]
fitness_aha = [1.1334817528148087, 0.8984674866788202, 0.8517543278994602, 0.8117543278994602, 0.8017543278994602, 0.7917543278994602, 0.7917543278994602, 0.7717543278994602, 0.7617543278994602, 0.7617543278994602, 0.7617543278994602, 0.7617543278994602, 0.7617543278994602, 0.7617543278994602, 0.76123456664324,  0.76123456664324,  0.76023456664324, 0.7499346457685324, 0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324,0.7499346457685324]
fitness_random = [1.0751564201420782, 1.112321173387201, 1.0666864572736636, 1.068609135709203, 1.050616685135132, 1.0798152770697271, 1.0621358988688505, 1.037578999030208, 1.06334516733352, 1.0316628480968955, 1.0771680223182216, 1.0716539040048279, 1.0549058255086932, 1.035726031024496, 1.0728682991112035, 1.0321077600074189, 1.1494093036442522, 1.0598723837680482, 1.0600180127701442, 1.0781952606294776, 1.0363947795812496, 1.0440740750791693, 1.0422691924425618, 1.034059972413066, 1.048325547977007, 1.031819757481262, 1.043658420904575, 1.0442569712651122, 1.0542954086173848, 1.0591585933218668, 1.0744603268836848, 1.0346791839681997, 1.0557132477521767, 1.0348469731356482, 1.0315750777725525, 1.0547807298807097, 1.051098760351526, 1.0649518799289157, 1.0757368221891044, 1.038999549457864, 1.0551491455880628, 1.0407599709320188, 1.0534091441912037, 1.0590154103966358, 1.065946668736041, 1.0428394016314841, 1.04668531463355, 1.0397518348916301, 1.1612257804325809, 1.0696345230546227, 1.0480607196320673, 1.1443461772507179, 1.0394441584246827, 1.0337851101507882, 1.0619652471748937, 1.068536142164048, 1.0555890084000055, 1.0442438788780677, 1.0364878512261597, 1.0590645326550314, 1.036582504355543, 1.0437823165479432, 1.0363364802919977, 1.0572653349402155, 1.0557900166227139, 1.0403791785479617, 1.0636279418712955, 1.0646331482694298, 1.068188674166642, 1.0374283395931334, 1.0753882713433485, 1.0385517467275722, 1.0478588784638143, 1.0628587108381844, 1.0384896270093507, 1.054625578365427, 1.0541681010562383, 1.0425534386370001, 1.0559979679518363, 1.0476792111592326, 1.0733482765185234, 1.069619999688046, 1.046285964259317, 1.144470533963504, 1.0681904197206065, 1.067715198386931, 1.0590496236266758, 1.04184674672556, 1.0422579679965789, 1.0348739352319678, 1.0795550951595694, 1.041358284458702, 1.0473309597661953, 1.0543503175254245, 1.0475769964827217, 1.053208934984423, 1.035832265265648, 1.0344695400521655, 1.036528606745758, 1.0729897493164295]

fitness_gabpso= [1.083869030299626, 0.9773774553863004, 0.735912573651561, 0.7006645583434039, 0.7006645583434039, 0.7006645583434039, 0.6969320442017928, 0.6969320442017928, 0.6969320442017928, 0.6960153412667602,0.6839346457685324,0.6839346457685324,0.6839346457685324,0.6839346457685324,0.6839346457685324,0.6839346457685324,0.6839346457685324,0.6839346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324, 0.6739346457685324, 0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324,0.6739346457685324]

fitness_psoplus = [1.0716151303813501, 0.8543008699445884, 0.8543008699445884, 0.7631353596846542, 0.7631353596846542, 0.7631353596846542, 0.7631353596846542, 0.7631353596846542, 0.7505489643471592, 0.7505489643471592, 0.7505489643471592, 0.7505489643471592, 0.6753211666401924, 0.6753211666401924, 0.6753211666401924, 0.6753211666401924, 0.6753211666401924, 0.6753211666401924, 0.6753211666401924, 0.6753211666401924, 0.6481990936333257, 0.6481990936333257, 0.6481990936333257, 0.6481990936333257, 0.6481990936333257, 0.6481990936333257, 0.6384990936333257, 0.6384900913761367, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377, 0.6384326180399377]



# 绘制折线图

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.figure(figsize=(100, 6))
plt.plot(x, fitness_random, marker='x', linestyle='-', label='RANDOM',color='blue',markevery=5)
plt.plot(x, fitness_ga, marker='s', linestyle='-', label='GA',color='red',markevery=5)
plt.plot(x, fitness_aha, marker='*', linestyle='-', label='AHA',color='green',markevery=5)
plt.plot(x, fitness_pso, marker='o', linestyle='-', label='PSO',color='black',markevery=5)
plt.plot(x, fitness_gabpso, marker='+', linestyle='-', label='GA-BPSO',color='orange',markevery=5)
plt.plot(x, fitness_psoplus, marker='v', linestyle='-', label='IPSO',color='purple',markevery=5)
# 添加标题和轴标签
plt.xlabel('迭代次数')
plt.ylabel('适应度值')

# 显示图例
plt.savefig('three.png')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.55))

# 显示图表
plt.show()