# DDT-Net

<img src="https://github.com/Moriartest/DDT-Net/blob/master/ReadMe_IMG/ddt4.png?raw=true" width="500" height="400">

### Environment
+ Python 3.9
  

### Requirements
Please `pip install` the following packages:
+ torch>=2.1.0
+ torchvision>=0.6.1
+ yacs==0.1.8
+ numpy>=1.19.5
+ opencv
+ torchvision>=0.16.0
+ scikit-learn==0.24.1


### Usage
#### Dataset
##### Training sets

+ [DEFACTO]
+ [DIS25k]
+ [IMD2020]
+ [MS-COCO]

##### Test sets

+ [DEFACTO]
+ [Columbia]
+ [DIS25k]
+ [CASIAv1]
+ [IMD2020]
+ [MS-COCO]


## Inference
### step 1: Install python packages in [requirement.txt](https://github.com/Moriartest/DDT-Net/blob/master/requirements.txt) .

### step 2: Download the weight `output/loss-0.0164.pkl` to the root directory.

- Model weights and test results download link：[百度网盘](https://pan.baidu.com/s/1tj-s5aSEqT6MHrASyJmmVA?pwd=DDTN) (提取码：DDTN)

### step 3:Modify the prediction txt image file: data/pred.txt

### step 4: Run the following script to obtain detection results in the testing image.
  `python predict.py`
  
### step 5: Run the following script to evaluate
 `python test.py`
 
__Note: The input image size can be modified in line 64 of predict.py. In this example, the image is input at 640*640.

<table>
  <tr>
    <th rowspan="2">方法</th>
    <th colspan="4">DEFACTO</th>
    <th colspan="4">DIS25k</th>
    <th colspan="4">IMD</th>
  </tr>
  <tr>
    <th>F1</th>
    <th>AUC</th>
    <th>Sen.</th>
    <th>Spe.</th>
    <th>F1</th>
    <th>AUC</th>
    <th>Sen.</th>
    <th>Spe.</th>
    <th>F1</th>
    <th>AUC</th>
    <th>Sen.</th>
    <th>Spe.</th>
  </tr>
  <tr>
    <td>MVSS-Net</td>
    <td>35.62</td>
    <td>0.62</td>
    <td>85.14</td>
    <td>22.52</td>
    <td>34.14</td>
    <td>0.69</td>
    <td>83.72</td>
    <td>21.44</td>
    <td>40.92</td>
    <td>0.67</td>
    <td>88.76</td>
    <td>26.59</td>
  </tr>
  <tr>
    <td>NEDB-Net</td>
    <td>66.67</td>
    <td>0.53</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.67</td>
    <td>0.54</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.67</td>
    <td>0.58</td>
    <td>100.0</td>
    <td>0.0</td>
  </tr>
  <tr>
    <td>RRU-Net</td>
    <td>66.67</td>
    <td>0.95</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.67</td>
    <td>0.95</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.67</td>
    <td>0.93</td>
    <td>100.0</td>
    <td>0.0</td>
  </tr>
  <tr>
    <td>PSCC-Net</td>
    <td>64.29</td>
    <td>0.63</td>
    <td>63.0</td>
    <td>67.0</td>
    <td>22.3</td>
    <td>0.17</td>
    <td>17.04</td>
    <td>64.2</td>
    <td>56.19</td>
    <td>0.47</td>
    <td>47.57</td>
    <td>78.28</td>
  </tr>
  <tr>
    <td>DDT-Net (Ous)</td>
    <td><b>89.18</b></td>
    <td>0.89</td>
    <td>91.24</td>
    <td>86.62</td>
    <td><b>93.55</b></td>
    <td>0.99</td>
    <td>98.92</td>
    <td>87.44</td>
    <td><b>67.83</b></td>
    <td>0.95</td>
    <td>94.76</td>
    <td>15.36</td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th rowspan="2">方法</th>
    <th colspan="4">CASIAv1</th>
    <th colspan="4">Columbia</th>
    <th colspan="4">NIST16</th>
  </tr>
  <tr>
    <th>F1</th>
    <th>AUC</th>
    <th>Sen.</th>
    <th>Spe.</th>
    <th>F1</th>
    <th>AUC</th>
    <th>Sen.</th>
    <th>Spe.</th>
    <th>F1</th>
    <th>AUC</th>
    <th>Sen.</th>
    <th>Spe.</th>
  </tr>
  <tr>
    <td>MVSS-Net</td>
    <td>75.34</td>
    <td>0.84</td>
    <td>61.55</td>
    <td>97.11</td>
    <td>50.0</td>
    <td>0.98</td>
    <td>100.0</td>
    <td>33.0</td>
    <td>17.72</td>
    <td>0.67</td>
    <td>100.0</td>
    <td>9.72</td>
  </tr>
  <tr>
    <td>NEDB-Net</td>
    <td>69.71</td>
    <td>0.77</td>
    <td>99.78</td>
    <td>0.25</td>
    <td>66.67</td>
    <td>0.62</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.67</td>
    <td>0.52</td>
    <td>100.0</td>
    <td>0.0</td>
  </tr>
  <tr>
    <td>RRU-Net</td>
    <td>70.13</td>
    <td>0.92</td>
    <td>91.83</td>
    <td>19.22</td>
    <td>66.67</td>
    <td>0.92</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.67</td>
    <td>0.9</td>
    <td>100.0</td>
    <td>0.0</td>
  </tr>
  <tr>
    <td>PSCC-Net</td>
    <td>73.84</td>
    <td>0.67</td>
    <td>66.88</td>
    <td>83.54</td>
    <td>66.79</td>
    <td>0.75</td>
    <td>100.0</td>
    <td>0.56</td>
    <td>67.65</td>
    <td>0.9</td>
    <td>95.14</td>
    <td>13.89</td>
  </tr>
  <tr>
    <td>DDT-Net (Ous)</td>
    <td><b>75.92</b></td>
    <td>0.88</td>
    <td>87.79</td>
    <td>50.13</td>
    <td>66.3</td>
    <td>0.53</td>
    <td>100.0</td>
    <td>0.0</td>
    <td>66.33</td>
    <td>0.9</td>
    <td>90.28</td>
    <td>18.06</td>
  </tr>
</table>


### Contact
+ Jim Wong (202208540021083@ctgu.edu.cn)
