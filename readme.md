<!--
 * @Date: 2024-03-22 22:39:53
 * @LastEditors: wurh2022 z02014268@stu.ahu.edu.cn
 * @LastEditTime: 2024-04-08 00:12:37
 * @FilePath: \Bearing_prediction\readme.md
 * @Description:  
-->
# 本科毕业设计项目

---

## 项目综述





## 实现过程

#### 2024年4月7日
> 完成了基本的数据集的处理，但是在搭建模型时遇到问题，torch的transformer模型怎么用？最终选择了最基础的transformer模型，使用到编码器部分，解码器用一个全连接层代替
#### 2024年4月8日
> 尝试写模型的前向传播和训练代码，对于epoch、step、batch_size等参数有了新的认识，一般来说使用epoch来进行迭代训练，每个epoch包含多个step，每个step包含多个batch_size，每个batch_size包含多个样本，每个样本包含多个特征。一个batch可以批量的输入到模型，这是不需要迭代的
### 数据处理



绘制一个5行11列的表格，使用HTML实现
    
    
<table border="1">
<tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
</tr>
<tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
</tr>
<tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
</tr>
<tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
</tr>
<tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
    <td>11</td>
</tr>

<table>
<capital>如何在Markdown里面画这样的表格：</capital>
<tr>
<th>普通表头</th>
<th align="right"><i>斜体表头而且居右</th>
<th colspan=2>表头横向合并单元格</th>
<td width="80px">限制列宽为80px超出会自动换行</td>
</tr>
<tr>
<th>左边也可以有表头</th>
<td bgcolor=#ffffcc>涂个颜色</td>
<td><mark>高亮文本</mark>但不全高亮</td>
<td><b>有时候加粗</b><i>有时候斜体</i></td>
<td width="20px">20px小于80px服从80px列宽命令无效</td>
</tr>
<tr>
<td>表头不一定是一整行或者一整列的</td>
<td rowspan=2>纵向合并单元格要注意<br>下一行少一个单元格<br>字太多必要时我会换行</td>
<td rowspan=2 colspan=2>单元格也可以从两个方向合并</td>
<td rowspan=2 width="10%">百分比和像素是可以混用的具体服从哪个取决于哪个大</td>
</tr>
<td align="left"> 简单做个居左 </td>
</tr>
</table>

