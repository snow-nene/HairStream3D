可以。我把这篇 **HairLRM** 压缩成一份偏“**复现/读代码用的技术笔记**”：不按论文逐段复述，而是把**核心思想 → 数据 → 网络 → 公式 → 训练 → 推理 → 2D→3D → 后处理 → strand extraction → refinement → 超参数**串成一条完整链路。以下内容只依据你上传的论文。论文的根本出发点是：不要直接从单张 2D 图像回归高维 3D strand field，而是先让 LRM 给一个可靠的 3D surface scaffold，再在一个学习到的 hair latent space 中反演 orientation field。

---

# HairLRM 一句话版本

整个 HairLRM 可以压缩成：

$$
\boxed{
I
\xrightarrow{\text{LRM}}
M
\xrightarrow[\text{2D HairStep constraints}]{\text{optimize latent }z}
O^{3D}
\xrightarrow{\text{surface correction}}
O^*
\xrightarrow{\text{trace/grow}}
H
\xrightarrow{\text{re-encode + optimize}}
H_{\rm final}
}
$$

其中真正最重要的不是 Hunyuan3D，也不是 HairStep，而是：

$$
\boxed{\text{Mesh prior}+\text{DOAE learned prior}+\text{2D observation}}
$$

三者一起解决 3D hair reconstruction。

论文指出传统方法的问题本质上有两个：**Global Structural Collapse**，即单视图无法可靠判断马尾等结构的深度/遮挡；以及 **Manifold Ambiguity**，单一 orientation field 容易把卷发、多方向流场过度平滑。HairLRM 用 LRM mesh 解决前者，用 dual point-cloud latent representation + orientation field 解决后者。

---

# 1. 整个 Pipeline

论文 Fig. 2 基本就是：

```text
                 ┌──────────────┐
Input image I ──►│ Image-to-3D  │
                 │ LRM          │
                 └──────┬───────┘
                        │
                        ▼
                   3D Mesh M
                        │
          ┌─────────────┴──────────────┐
          │                            │
          ▼                            ▼
   SDF supervision              Rasterize M → image
                                       │
Input image ──► HairStep ──────────────┤
              │                        │
              ├─ 2D orientation        ▼
              └─ hair mask        2D → 3D lifting
                                       │
                                 O_proj, C_proj
                                       │
                                       ▼
                         optimize latent code z
                                       │
                                       ▼
                           Fixed DOAE Decoder
                                       │
                      ┌────────────────┼─────────────┐
                      ▼                ▼             ▼
                     SDF         Orientation       Class
                      D               O             C
                                      │
                                      ▼
                         Surface Normal Correction
                                      │
                                      ▼
                          Corrected Field O*
                                      │
                                      ▼
                      CT2Hair + Newton Advection
                                      │
                                      ▼
                              Hair Strands H
                                      │
                                      ▼
                           synthesize point clouds
                                      │
                                      ▼
                             DOAE Encoder
                                      │
                                      ▼
                              refined z*
                                      │
                                      ▼
                          optimize z* again
                                      │
                                      ▼
                           FINAL STRANDS
```

DOAE 的任务就是把 LRM 提供的粗 surface geometry 转成高质量 volumetric 3D orientation field，然后从这个 field 里长出 strands。

---

# 2. 最关键的 Representation

这篇论文实际上没有直接让网络：

$$
I\rightarrow \{\text{hair strands}\}
$$

而是学习：

$$
\boxed{
z\rightarrow
\left(
D(\mathbf x),
O(\mathbf x),
C(\mathbf x)
\right)
}
$$

对于任意 3D query：

$$
\mathbf x\in\mathbb R^3
$$

decoder 输出三个东西：

### SDF

$$
D(\mathbf x)\in\mathbb R
$$

表示 mesh surface。

### Orientation

$$
O(\mathbf x)\in\mathbb R^3
$$

表示这个空间位置的 hair flow direction。

### Classification

$$
C(\mathbf x)
$$

判断：

```text
hair
vs
bust/non-hair
```

所以 HairLRM 的核心中间表示其实是：

$$
\boxed{
\text{implicit geometry}
+
\text{implicit vector field}
+
\text{semantic field}
}
$$

而不是 strands。

每一个训练 sample：

$$
\boxed{
s=[\mathbf x,\mathbf o,c]\in\mathbb R^7
}
$$

分别为：

$$
3D\ position+3D\ orientation+1\ binary\ class.
$$



---

# 3. DOAE：Dual Orientation AutoEncoder

这是论文真正的核心网络。

它借鉴 Dora / 3DShape2VecSet VAE，但输入不是普通 surface point cloud。

作者设计两个 point cloud：

$$
P_{\rm uniform}
$$

和

$$
P_{\rm salient}.
$$

其中：

**Uniform branch**

$$
P_{\rm uniform}
$$

负责：

$$
\boxed{\text{global hair volume}}
$$

也就是头发整体在哪里、多大、轮廓如何。

**Salient branch**

$$
P_{\rm salient}
$$

专门强调：

* curls
* partings
* flow boundaries
* strong directional changes

即：

$$
\boxed{\text{high-frequency orientation}}
$$

这就是论文所谓的 **Dual**，不是“两套 orientation vector”，而是**两个互补 point-cloud streams**。

---

# 4. Encoder

先分别做 FPS：

$$
P_{\rm uniform},P_{\rm salient}
\xrightarrow{\rm FPS}
P_{\rm sparse}.
$$

然后：

$$
C_{\rm uniform}
=
\operatorname{CrossAttn}
(
\operatorname{PosEmb}(P_{\rm sparse}),
\operatorname{PosEmb}(P_{\rm uniform})
)
$$

以及：

$$
C_{\rm salient}
=
\operatorname{CrossAttn}
(
\operatorname{PosEmb}(P_{\rm sparse}),
\operatorname{PosEmb}(P_{\rm salient})
).
$$

这是 Eq. (1)。

然后融合：

$$
\boxed{
z=
\operatorname{SelfAttn}
(C_{\rm uniform}+C_{\rm salient})
}
\tag{2}
$$

所以可以把 encoder 理解成：

```python
def encode(P_uniform, P_salient):

    P_sparse = FPS(...)

    C_uniform = cross_attention(
        pos_emb(P_sparse),
        pos_emb(P_uniform)
    )

    C_salient = cross_attention(
        pos_emb(P_sparse),
        pos_emb(P_salient)
    )

    z = self_attention(
        C_uniform + C_salient
    )

    return z
```

关键思想：

$$
\boxed{
z =
global\ shape
+
local\ directional\ details
}
$$

消融也证实两支缺一不可：Uniform-only 和 Salient-only 的总 MAE 分别达到 \(12.78^\circ\) 和 \(13.37^\circ\)，完整模型是 \(7.13^\circ\)。

---

# 5. Decoder

给任意：

$$
P_{\rm query}
=
\{\mathbf x_1,\ldots,\mathbf x_N\}
$$

decoder 用 latent \(z\) 查询三个 field：

$$
\hat D
=
\operatorname{CrossAttn}_D
(
\operatorname{SelfAttn}_D(z),
\operatorname{PosEmb}(P_{\rm query})
)
$$

$$
\hat O
=
\operatorname{CrossAttn}_O
(
\operatorname{SelfAttn}_O(z),
\operatorname{PosEmb}(P_{\rm query})
)
$$

$$
\hat C
=
\operatorname{SoftMax}
\left[
\operatorname{CrossAttn}_C
(
\operatorname{SelfAttn}_C(z),
\operatorname{PosEmb}(P_{\rm query})
)
\right].
\tag{3}
$$

这里 D/O/C 三个 decoder branch 使用不同参数。

代码思维就是：

```python
def decode(z, query_xyz):

    q = positional_embedding(query_xyz)

    D = cross_attn_D(
        self_attn_D(z), q
    )

    O = cross_attn_O(
        self_attn_O(z), q
    )

    C_logits = cross_attn_C(
        self_attn_C(z), q
    )

    C = softmax(C_logits)

    return D, O, C
```

---

# 6. 训练数据怎么造

这个部分非常重要，因为 DOAE 的能力主要来自这里。

作者收集：

$$
52K
$$

个 strand hairstyles：

$$
40K\ \text{DiffLocks}
+
10K\ \text{Perm}
+
2K\ \text{UniHair}.
$$

包括 straight / wavy / curly / long / braided 等。每个 hairstyle 先平移和统一缩放，对齐 canonical bust。

原始 hair：

$$
H=\{S_i\}
$$

其中 strand：

$$
S_i=\{\mathbf x_{ij}\}.
$$

由 strand tangent 得：

$$
\mathbf o_{ij}.
$$

于是：

$$
P_{\rm dense}
=
\{
[\mathbf x_{ij},\mathbf o_{ij},1]
\}.
$$

---

# 7. 从 strands 制造 surface mesh

先把 dense strands voxelize 成 occupancy grid。

然后：

```text
hair strands
    ↓
densify
    ↓
occupancy grid
    ↓
iso-surface
    ↓
M_hair
```

再与 bust 做 boolean union：

$$
\boxed{
M_{\rm surface}
=
M_{\rm hair}\cup M_{\rm bust}
}
$$

这个 surface 用来产生训练 SDF：

$$
D(\mathbf x)=
\operatorname{SDF}(\mathbf x,M_{\rm surface}).
$$



---

# 8. Salient point cloud 怎么造

这个设计很有意思。

普通 mesh 方法会找 sharp edges，但 hair 的核心不是 triangle sharpness，而是 **strand-flow change**。

所以作者先对 strands 做 k-means。

两个 strands：

$$
S_i,S_j
$$

的距离：

$$
\boxed{
\gamma(S_i,S_j)
=
\frac1{N_s}
\sum_{k=1}^{N_s}
\|\mathbf x_{ik}-\mathbf x_{jk}\|_2^2
}
$$

然后分成：

$$
N_{\rm cluster}=256
$$

clusters。

每个 cluster 找最接近 cluster center 的 strand：

```python
for cluster in clusters:
    salient_strand = nearest_strand_to_center(cluster)
```

再在这些 salient strands 上均匀 sample。

最终：

$$
P_{\rm salient}.
$$

实际：

$$
\boxed{N_{\rm salient}=50K}
$$

points。训练参数见 Appendix A。

---

# 9. Uniform point cloud 怎么造

构造整个 hairstyle bounding box。

论文用 cell size：

$$
0.2^3\ {\rm mm}^3
$$

的 uniform grid。

每个 candidate point：

$$
\mathbf x
$$

找最近 dense strand point：

$$
j^*
=
\arg\min_j
\|\mathbf x-\mathbf x_j\|.
$$

然后 orientation：

$$
O(\mathbf x)=O(\mathbf x_{j^*}).
$$

距离过远：

$$
\|\mathbf x-\mathbf x_{j^*}\|>\epsilon_d
$$

则删除。

此外在 bust volume 内采样：

$$
P_{\rm bust}.
$$

最终：

$$
\boxed{
P_{\rm uniform}
=
P_{\rm bust}\cup P_{\rm uniform}^*
}
$$

hair：

$$
c=1
$$

bust：

$$
c=0,\qquad O=(0,0,0).
$$

最终每个 hairstyle 的 uniform points 数量：

$$
\boxed{
500K\sim2M
}
$$

短发约 500K，长发可到 2M。

---

# 10. Orientation Even Sampling

这是一个很容易被忽略、但实际非常重要的设计。

hair dataset 的 orientation distribution 严重不均衡。

例如大量：

$$
(0,-1,0)
$$

这种向下的直发。

如果 random sampling，network 会天然倾向 straight dominant directions。

作者在单位球面上用 Fibonacci sphere 产生：

$$
N_{\rm dir}
$$

个均匀方向 anchor：

$$
\{\mathbf a_k\}_{k=1}^{N_{\rm dir}}.
$$

每个 orientation：

$$
\mathbf o_i
$$

分配：

$$
k^*
=
\arg\max_k
\mathbf o_i^\top\mathbf a_k.
$$

于是 orientation sphere 被划成 bins。

然后：

$$
\boxed{\text{从每个 direction bin 等量 sampling}}
$$

而不是从所有 point random sample。

此外 query point 还使用基于 surface SDF 的 spatial importance sampling。

这一步对 curly hair 很重要；论文消融中 even sampling 使 curly MAE 大约降低 \(4.5^\circ\)。

---

# 11. DOAE Training Loss

完整 training objective：

$$
\boxed{
L_{\rm train}(\psi)
=
\lambda_D
\|D-\hat D\|_2^2
+
\lambda_O
\|O-\hat O\|_2^2
+
\lambda_C
L_{\rm CE}(\hat C,C)
+
\lambda_{\rm KL}L_{\rm KL}
}
$$

其中：

$$
D=\text{GT SDF}
$$

$$
O=\text{nearest dense strand orientation}
$$

$$
C=\text{hair/bust label}.
$$

因为是 VAE latent：

$$
L_{\rm KL}
=
D_{\rm KL}
(
q(z|P)\|N(0,I)
).
$$

论文实际 loss weights：

$$
\boxed{
\lambda_D=10
}
$$

$$
\boxed{
\lambda_O=0.02
}
$$

$$
\boxed{
\lambda_C=0.01
}
$$

$$
\boxed{
\lambda_{\rm KL}=0.001
}
$$

KL 只在 training 使用。

---

# 12. Training 配置

Appendix 给得很明确：

```text
GPU:       4 × high-end NVIDIA GPU
VRAM:      96 GB / GPU class described
Batch:     32
Optimizer: Adam
LR:        1e-4

query points:
100K / hairstyle / iteration

iterations:
300K

training time:
~72 hours
```



这说明 DOAE 本身是一个相当重的 pretrained hair prior。

---

# 13. 推理阶段最重要的变化

这里非常关键：

> 推理的时候基本**不训练 DOAE network**。

训练好的：

```text
Encoder
Decoder
```

固定。

真正优化的是：

$$
\boxed{z}
$$

也就是 latent inversion。

---

# 14. Input Image → LRM Mesh

输入：

$$
I_{\rm original}.
$$

先 segment hair，然后把 portrait compositing 到 canonical template bust，产生标准化：

$$
I.
$$

然后：

$$
\boxed{
M_{\rm input}
=
\operatorname{LRM}(I)
}
$$

实验中用：

$$
\boxed{\text{Hunyuan3D}}
$$

但论文强调：

$$
\text{DOAE only consumes }M
$$

所以理论上 LRM backbone 可以换。

---

# 15. Runtime latent optimization

初始化：

$$
\boxed{
z_0=\bar z_{\rm train}
}
$$

也就是 training-set latent mean。

而不是 random latent。

然后 decoder 固定：

$$
(\hat D,\hat O,\hat C)
=
F_\theta(z,P_{\rm query}).
$$

runtime optimization：

$$
\boxed{
L_{\rm opt}(z)
=
\lambda_D
\|D-\hat D\|_2^2
+
\lambda_O
\|\hat O-O_{\rm proj}\|_2^2
+
\lambda_C
L_{\rm CE}(\hat C,C_{\rm proj})
}
\tag{4}
$$

这三个 supervision 分别来自：

```text
LRM mesh     → D
input image  → O_proj
input image  → C_proj
```

也就是：

$$
\boxed{
\text{3D shape constraint}
+
\text{2D hair-flow constraint}
+
\text{2D silhouette constraint}
}
$$

共同反演 latent。

---

# 16. 为什么 SDF loss 特别重要

对 LRM mesh：

$$
M_{\rm input}
$$

随机 sample 3D query points：

$$
\mathbf x_i.
$$

计算：

$$
D_i
=
\operatorname{SDF}
(
\mathbf x_i,M_{\rm input}
).
$$

然后要求：

$$
\hat D(z,\mathbf x_i)\approx D_i.
$$

所以：

$$
z
$$

不能随便生成一个 plausible hairstyle。

它必须对应：

$$
\boxed{\text{LRM mesh 的 global 3D geometry}}
$$

这正是 HairLRM 比纯 image→hair prior 更稳定的原因。

---

# 17. 2D HairStep supervision

HairStep 从图像预测：

$$
O_{\rm screen}
$$

和：

$$
C_{\rm screen}.
$$

但 appendix 给出了比正文更完整的 segmentation：

SAM 分别得到：

$$
C_{\rm input}
$$

和 mesh rendered view：

$$
C_{\rm render}.
$$

真正 validity mask：

$$
\boxed{
C_{\rm screen}
=
C_{\rm input}\cap C_{\rm render}
}
$$

这样只保留：

> image 认为是 hair，同时当前 3D mesh projection 也认为是 hair

的区域。

这会过滤 background / occlusion / mismatch。

---

# 18. 论文真正的 2D → 3D Orientation Lifting

这部分非常值得注意，因为不是简单的 barycentric backprojection。

先 rasterize mesh：

$$
M\rightarrow I_{p2f}.
$$

其中：

$$
\boxed{
I_{p2f}(u,v)=f_i
}
$$

表示 pixel：

$$
(u,v)
$$

对应 visible mesh triangle：

$$
f_i.
$$

作者说可以用 PyTorch3D 实现。

---

# 19. Differential Vector Lifting

HairStep 给：

$$
o_i
=
O_{\rm screen}(u_i,v_i)
\in\mathbb R^2.
$$

先 normalize：

$$
\boxed{
\hat o_i
=
-\frac{o_i}{\|o_i\|_2}
}
$$

然后沿 2D orientation 走：

$$
\boxed{
p_{\rm next}
=
\left(
\lfloor u_i+\delta\hat o_{u,i}\rfloor,
\lfloor v_i+\delta\hat o_{v,i}\rfloor
\right)
}
\tag{6}
$$

论文：

$$
\boxed{\delta=2\ pixels}
$$



然后 rasterization map 给：

$$
p_i\rightarrow f_i
$$

$$
p_{\rm next}\rightarrow f_{\rm next}.
$$

取两个 triangle 的 3D centroid：

$$
P_i
$$

和：

$$
P_{\rm next}.
$$

于是：

$$
\boxed{
O_{\rm proj}
=
\operatorname{normalize}
(P_{\rm next}-P_i)
}
\tag{7}
$$



这就是论文解决：

$$
2D\ direction\not\Rightarrow unique\ 3D\ direction
$$

的方法。

不是：

$$
(o_x,o_y)\rightarrow(o_x,o_y,0)
$$

而是：

$$
\boxed{
\text{沿 image orientation 移动一点}
\rightarrow
\text{分别查询 surface 3D point}
\rightarrow
\text{两 3D points 作差}
}
$$

非常简单，但很关键。

代码就是：

```python
def lift_orientation(pixel, ori2d, pix_to_face, mesh):

    o = -ori2d / norm(ori2d)

    p0 = pixel

    p1 = floor(
        pixel + 2.0 * o
    )

    f0 = pix_to_face[p0]
    f1 = pix_to_face[p1]

    P0 = centroid(mesh.faces[f0])
    P1 = centroid(mesh.faces[f1])

    ori3d = normalize(P1 - P0)

    return ori3d
```

---

# 20. Mesh fitting / camera alignment

Appendix 还补了一套 coarse-to-fine registration。

首先多视角 render LRM mesh。

MediaPipe：

$$
\{P^v\}
$$

预测每个 view 的 2D facial landmarks。

已知：

$$
K,\ [R|T]
$$

进行 triangulation：

$$
\{P^v\}
\rightarrow
P_{\rm face}^{3D}.
$$

然后 RANSAC：

$$
P_{\rm face}
\leftrightarrow
P_{\rm tgt}
$$

估计：

$$
T_{\rm coarse}.
$$

接着 region-aware ICP。

作者没有直接拿整个 hair mesh ICP，因为 hair shape 本来就变化很大；而是使用：

```text
stable body below head
+
high-confidence face landmarks
```

作为 registration source。

最终：

$$
\boxed{
T_{\rm final}
=
T_{\rm refine}T_{\rm coarse}
}
$$

再作用到完整 mesh。

---

# 21. Latent Optimization 代码本质

整个 inference 的核心其实可以缩成：

```python
decoder.requires_grad_(False)

z = mean_training_latent.clone()
z.requires_grad_(True)

optimizer = Adam([z])

for step in range(300):

    xyz = sample_query_points(100_000)

    D_gt = sdf(mesh, xyz)

    O_proj, C_proj = image_constraints(...)

    D_pred, O_pred, C_pred = decoder(z, xyz)

    loss = (
          10.0 * mse(D_pred, D_gt)
        + 0.02 * mse(O_pred, O_proj)
        + 0.01 * cross_entropy(C_pred, C_proj)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

论文推理时：

$$
\boxed{100K\ query\ points/iteration}
$$

以及：

$$
\boxed{300\ iterations}
$$

第一轮和 refinement 都如此。

---

# 22. Dense Orientation Field

优化得到：

$$
z'
$$

之后不再只 sample 100K。

而是在 dense 3D grid：

$$
512^3
$$

上 query decoder：

$$
(\hat D,\hat O,\hat C)
=
F(z',P_{512^3}).
$$

最终 orientation：

$$
\boxed{
\tilde O
=
\hat O
\odot
\arg\max(\hat C)
\odot
\Omega
}
\tag{5}
$$

其中：

$$
\arg\max(\hat C)
$$

是 hair mask。

而：

$$
\Omega
$$

是：

$$
M_{\rm input}
$$

SDF 得到的 interior occupancy mask。

因此 orientation 只留下：

```text
decoder says hair
AND
inside mesh
```

的区域。

---

# 23. Surface-guided Orientation Correction

虽然 latent field 已经不错，但它没有硬性保证：

$$
O(\mathbf x)
$$

和 LRM mesh surface tangent 一致。

所以 strands 可能穿 mesh。

对于 orientation sample：

$$
s_i=[x_i,o_i,c_i]
$$

找 mesh closest point：

$$
\hat x_i
$$

以及 normal：

$$
\hat n_i.
$$

若：

$$
d(\hat x_i,x_i)>\epsilon_d
$$

保持：

$$
o_i^*=o_i.
$$

否则把 orientation 投影到 tangent plane。

数学上就是：

$$
\boxed{
o_{\rm proj}
=
\frac{
o-(o\cdot\hat n)\hat n
}{
\|o-(o\cdot\hat n)\hat n\|
}
}
$$

PDF 的解析文本在这一行有排版丢失，但 Algorithm 1 的语义就是**去掉 normal component**。

---

# 24. 距离加权 blend

不能直接：

$$
o\leftarrow o_{\rm proj}
$$

否则 surface 附近和远处会产生突变。

因此：

$$
w
=
\frac{1}
{1+\alpha d(\hat x_i,x_i)}
$$

再：

$$
\boxed{
o^*
=
w o_{\rm proj}
+
(1-w)o
}
$$

最后：

$$
o^*
\leftarrow
\frac{o^*}{\|o^*\|}.
$$

论文：

$$
\boxed{
\epsilon_d=16{\rm mm}
}
$$

并通过 \(\alpha\) 约束 blend weight 到大约 \([0.1,1]\)。

代码：

```python
for x, o in orientation_samples:

    x_surface, n = closest_surface_point(mesh, x)
    d = norm(x - x_surface)

    if d > 16_mm:
        o_new = o

    else:
        o_tan = o - dot(o, n) * n
        o_tan = normalize(o_tan)

        w = 1.0 / (1.0 + alpha * d)

        o_new = normalize(
            w * o_tan +
            (1.0 - w) * o
        )
```

---

# 25. Orientation Field → Hair Strands

这里作者没有简单地：

> scalp 每个点直接沿 field 积分。

因为这样容易有 holes。

也没有纯 CT2Hair。

而是 hybrid extraction。

首先：

$$
O^*
$$

离散成 dense point cloud。

然后执行 CT2Hair pipeline：

```text
mean-shift filtering
        ↓
segment generation
        ↓
guided growing
        ↓
guide strands G
```

再检查 scalp vertices。

没有被 guide strands cover 的 scalp vertices：

$$
V_{\rm uncovered}
$$

从这里用：

$$
\boxed{\text{forward Newton advection}}
$$

继续沿：

$$
O^*
$$

trace strands：

$$
E.
$$

最终：

$$
\boxed{
H_{\rm extract}
=
G\cup E
}
$$



所以：

```python
G = CT2Hair(corrected_field)

uncovered = find_uncovered_scalp_vertices(G)

E = []

for root in uncovered:
    strand = newton_advection(
        root,
        corrected_field
    )
    E.append(strand)

H = G + E
```

注意：论文没有在正文/附录进一步展开 Newton advection 的具体数值更新式，所以如果严格按这篇论文，不能凭空补一个作者没有给出的 integration equation。

---

# 26. Iterative Refinement

这是第二个很重要的创新。

第一轮：

$$
z_0=\bar z
$$

然后：

$$
z_0
\xrightarrow{\text{optimization}}
z_1
$$

得到：

$$
O_1
\rightarrow
H_1.
$$

问题是：

$$
\bar z
$$

对 out-of-distribution hair 未必是好的初始化。

所以作者把自己生成的：

$$
H_1
$$

重新当成“训练 hairstyle”。

按照 Sec. 3.2 完整重新做：

$$
H_1
\rightarrow
P_{\rm uniform}^{(1)}
$$

和：

$$
H_1
\rightarrow
P_{\rm salient}^{(1)}.
$$

然后送回 encoder：

$$
\boxed{
z^*
=
E_\phi
(
P_{\rm uniform}^{(1)},
P_{\rm salient}^{(1)}
)
}
$$

再以：

$$
z^*
$$

为初始化重新优化 Eq. 4：

$$
z^*
\xrightarrow{300\ steps}
z_{\rm final}.
$$

论文发现：

$$
\boxed{\text{一次 refinement 足够}}
$$



---

# 27. 为什么 Refinement 有效

这里本质上是：

第一次：

$$
\bar z_{\rm train}
\rightarrow
\text{gradient search}
\rightarrow
z_1.
$$

但是 gradient search 可能卡 local minimum。

而第一次输出：

$$
H_1
$$

已经包含相当多 input-specific geometry。

所以重新 encode：

$$
H_1\rightarrow z^*
$$

相当于把 latent initialization 从：

$$
\text{generic mean hair}
$$

变成：

$$
\boxed{\text{current reconstructed hair prior}}
$$

然后再 optimization。

所以不是：

```text
600 gradient steps
```

而是：

```text
300 steps
→ reconstruct strands
→ encoder jump
→ 300 steps
```

论文专门做了公平消融：直接优化 600 次仍然容易卡 local minimum，而 300 + re-encode + 300 能明显改善 discontinuities 和 chaotic strands。

---

# 28. 完整 Training 伪代码

如果把整篇论文压成复现代码，大概是：

```python
# =====================================================
# DATASET
# =====================================================

for hair in hair_dataset:

    hair = align_to_template_bust(hair)

    # dense strand samples
    P_dense = []

    for strand in hair:
        for x in resample(strand):
            o = strand_tangent(x)
            P_dense.append([x, o, 1])

    # -------------------------------------------------
    # surface
    # -------------------------------------------------

    occupancy = voxelize(hair)

    M_hair = extract_isosurface(occupancy)

    M_surface = boolean_union(
        M_hair,
        M_bust
    )

    # -------------------------------------------------
    # salient stream
    # -------------------------------------------------

    clusters = strand_kmeans(
        hair,
        n_clusters=256
    )

    salient_strands = [
        nearest_strand_to_center(c)
        for c in clusters
    ]

    P_salient = sample_points(
        salient_strands,
        N=50_000
    )

    # -------------------------------------------------
    # uniform stream
    # -------------------------------------------------

    candidates = uniform_grid(
        global_bbox,
        cell_size=0.2  # mm scale from paper
    )

    P_uniform_hair = []

    for x in candidates:

        p = nearest(P_dense, x)

        if distance(x, p.xyz) < eps:
            P_uniform_hair.append(
                [x, p.orientation, 1]
            )

    P_bust = sample_bust_volume()

    P_uniform = (
        P_uniform_hair +
        P_bust
    )

    save(
        M_surface,
        P_uniform,
        P_salient,
        P_dense
    )
```

---

# 29. 完整 DOAE training

```python
for iteration in range(300_000):

    P_uniform, P_salient, M_surface, P_dense = batch()

    # -----------------------------
    # encoder
    # -----------------------------

    P_sparse = FPS(
        P_uniform,
        P_salient
    )

    C_u = CrossAttention(
        PosEmb(P_sparse),
        PosEmb(P_uniform)
    )

    C_s = CrossAttention(
        PosEmb(P_sparse),
        PosEmb(P_salient)
    )

    z = SelfAttention(
        C_u + C_s
    )

    # VAE latent sampling
    z = sample_latent(z)

    # -----------------------------
    # query sampling
    # -----------------------------

    P_query = orientation_even_sampling(
        P_dense,
        N=100_000
    )

    P_query = sdf_importance_sampling(
        P_query,
        M_surface
    )

    D_gt = sdf(M_surface, P_query)

    O_gt = nearest_orientation(
        P_dense,
        P_query
    )

    C_gt = nearest_class(
        P_dense,
        P_query
    )

    # -----------------------------
    # decoder
    # -----------------------------

    D_pred, O_pred, C_pred = decoder(
        z,
        P_query
    )

    # -----------------------------
    # loss
    # -----------------------------

    loss = (
        10.0 * mse(D_pred, D_gt)
        +
        0.02 * mse(O_pred, O_gt)
        +
        0.01 * CE(C_pred, C_gt)
        +
        0.001 * KL(z)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

# 30. 完整 Runtime 伪代码

把整篇 HairLRM 压缩到一个函数，基本就是：

```python
def HairLRM(image):

    # =================================================
    # 1. preprocessing
    # =================================================

    hair_mask = segment_hair(image)

    I = blend_with_template_bust(
        image,
        hair_mask
    )

    # =================================================
    # 2. LRM geometry prior
    # =================================================

    M = Hunyuan3D(I)

    M = fit_mesh_to_canonical_space(M)

    # =================================================
    # 3. 2D supervision
    # =================================================

    C_input = SAM(image)

    render = render_mesh(M)

    C_render = SAM(render)

    C_screen = (
        C_input & C_render
    )

    O_screen = HairStep(
        image * C_screen
    )

    # =================================================
    # 4. rasterization
    # =================================================

    pix_to_face = rasterize(M)

    O_proj = differential_lift(
        O_screen,
        pix_to_face,
        delta=2
    )

    C_proj = backproject_segmentation(
        C_screen,
        pix_to_face
    )

    # =================================================
    # 5. latent inversion
    # =================================================

    z = mean_training_latent()

    z = optimize_latent(
        z=z,
        mesh=M,
        O_proj=O_proj,
        C_proj=C_proj,
        iterations=300,
        query_points=100_000
    )

    # =================================================
    # 6. dense field
    # =================================================

    xyz = dense_grid(
        resolution=512
    )

    D, O, C = decoder(z, xyz)

    occupancy = inside_mesh(
        M, xyz
    )

    O = (
        O
        * argmax(C)
        * occupancy
    )

    # =================================================
    # 7. surface correction
    # =================================================

    O_corrected = surface_tangent_correction(
        O,
        M,
        eps=16_mm
    )

    # =================================================
    # 8. strands
    # =================================================

    guides = CT2Hair(
        O_corrected
    )

    uncovered_roots = (
        find_uncovered_scalp_vertices(
            guides
        )
    )

    extra = NewtonAdvection(
        uncovered_roots,
        O_corrected
    )

    H1 = guides + extra

    # =================================================
    # 9. iterative refinement
    # =================================================

    P_uniform, P_salient = synthesize_points(
        H1
    )

    z_refined = DOAE_encoder(
        P_uniform,
        P_salient
    )

    z_final = optimize_latent(
        z=z_refined,
        mesh=M,
        O_proj=O_proj,
        C_proj=C_proj,
        iterations=300
    )

    # =================================================
    # 10. final decode / correction / extraction
    # =================================================

    O_final = decode_dense_orientation(
        z_final
    )

    O_final = surface_tangent_correction(
        O_final,
        M
    )

    H_final = extract_strands(
        O_final
    )

    return H_final
```

---

# 31. Optional Manual Labeling

论文其实还支持人工控制。

用户直接在 rendered mesh 上画：

$$
S=\{p_1,p_2,\ldots,p_k\}.
$$

相邻点计算 screen tangent：

$$
\boxed{
t_i
=
\operatorname{normalize}
(p_{i+1}-p_i)
}
$$

然后完全使用前面的 differential lifting：

$$
t_i^{2D}
\rightarrow
t_i^{3D}.
$$

这些点作为：

$$
P_{\rm man}.
$$

如果多个 view：

$$
\boxed{
P_{\rm man}
=
\{P^v\}_{v=1}^{N_{\rm view}}
}
$$

全部 aggregate。

然后直接作为 latent optimization 的 ground-truth orientation constraints。

因此 HairLRM 不只是 reconstruction system，也天然可以成为：

$$
\boxed{\text{interactive hair editing system}}
$$

---

# 32. Latent Space 还能插值

DOAE 学到的 latent 不是乱的。

两种 hairstyle：

$$
H_A,H_B
$$

encode：

$$
z_A,z_B.
$$

插值：

$$
\boxed{
z(\alpha)
=
(1-\alpha)z_A+\alpha z_B,
\qquad
\alpha\in[0,1]
}
\tag{8}
$$

然后：

$$
z(\alpha)
\rightarrow O_\alpha
\rightarrow H_\alpha.
$$

论文 Fig. 14 显示 orientation field 和最终 strands 都会平滑变化，说明 latent manifold 有较好的 hairstyle semantics。

---

# 33. 所有关键超参数汇总

最值得复现时直接抄走的是这些：

| 参数                                |        论文设置 |
| --------------------------------- | ----------: |
| Hair dataset                      |         52K |
| DiffLocks                         |         40K |
| Perm                              |         10K |
| UniHair                           |          2K |
| \(N_{\rm cluster}\)               |         256 |
| \(N_{\rm salient}\)               |         50K |
| \(N_{\rm uniform}\)               |     500K–2M |
| train query / hairstyle / iter    |        100K |
| runtime query / iter              |        100K |
| dense inference grid              |   \(512^3\) |
| optimizer                         |        Adam |
| LR                                | \(10^{-4}\) |
| train batch                       |          32 |
| train iterations                  |        300K |
| training GPUs                     |           4 |
| training time                     |       ~72 h |
| first latent inversion            |    300 iter |
| refinement inversion              |   ≤300 iter |
| \(\lambda_D\)                     |          10 |
| \(\lambda_O\)                     |        0.02 |
| \(\lambda_C\)                     |        0.01 |
| \(\lambda_{KL}\)                  |       0.001 |
| differential lifting \(\delta\)   |    2 pixels |
| surface correction \(\epsilon_d\) |       16 mm |
| inference VRAM                    |       ~4 GB |
| training VRAM                     |      ~80 GB |
| total inference                   |  ~10–15 min |

训练与 inference 参数由 Appendix A 给出。

论文报告的 runtime 分解大约是：latent optimization ~10 min、LRM mesh ~3 min、hair growing 1–2 min、refinement + preprocessing <35 s。

---

# 34. 真正应该记住的 5 个创新点

如果把整篇 14 页论文继续压缩，我认为真正有价值的是下面五件事。

**① 不直接 Image → Strands，而是 Image → LRM Mesh → Strands。**

即：

$$
I\rightarrow M
$$

先把单视图最难的 global 3D ambiguity 交给大型 3D reconstruction prior。

---

**② 不直接回归最终 hair，而是优化 learned latent orientation field。**

$$
z
\rightarrow
(D,O,C)
$$

然后：

$$
\boxed{
z^*
=
\arg\min_z
\left[
10L_{\rm mesh}
+
0.02L_{\rm orientation}
+
0.01L_{\rm segmentation}
\right]
}
$$

也就是：

> pretrained hair prior 负责“什么样的 3D hair 是合理的”，LRM mesh 负责“整体 3D 形状是什么”，HairStep 负责“照片里可见的头发具体往哪里流”。

这是整篇论文最核心的思想。

---

**③ Dual point clouds 把 global shape 和 high-frequency flow 分开建模。**

$$
P_{\rm uniform}
\rightarrow
global\ volume
$$

$$
P_{\rm salient}
\rightarrow
curls/parting/flow\ boundaries
$$

然后：

$$
z
=
SelfAttn(
C_{\rm uniform}+C_{\rm salient}
).
$$

---

**④ 2D orientation 不直接变 3D，而是利用 mesh differential lifting。**

这是非常漂亮的一步：

$$
p
\xrightarrow{o_{2D}}
p+\delta o_{2D}
$$

然后：

$$
p\rightarrow P
$$

$$
p+\delta o_{2D}
\rightarrow P'
$$

最终：

$$
\boxed{
o_{3D}
=
\frac{P'-P}{\|P'-P\|}
}
$$

所以 mesh 不仅是 geometry prior，同时还是：

$$
\boxed{\text{2D observation → 3D constraint 的桥梁}}
$$

---

**⑤ 第一轮结果重新 encode，作为第二轮 latent inversion 的初始化。**

$$
\bar z
\rightarrow
optimize
\rightarrow
H_1
\rightarrow
Encoder
\rightarrow
z^*
\rightarrow
optimize
\rightarrow
H_{\rm final}.
$$

这实际上是一个：

$$
\boxed{
\text{analysis-by-synthesis closed loop}
}
$$

而不是普通 feed-forward reconstruction。

---

# 35. 最终把 HairLRM 写成一个数学问题

整篇论文最终可以被抽象成下面这个问题。

给定输入图像：

$$
I
$$

首先得到：

$$
M=LRM(I).
$$

由 image：

$$
(O_{\rm screen},C_{\rm screen})
=
HairStep(I).
$$

借助：

$$
M
$$

lift：

$$
(O_{\rm screen},C_{\rm screen})
\xrightarrow{Rasterization}
(O_{\rm proj},C_{\rm proj}).
$$

然后寻找 hair latent：

$$
\boxed{
z^*
=
\arg\min_z
\left[
\lambda_D
\|
D_M-D_\theta(z)
\|^2
+
\lambda_O
\|
O_{\rm proj}-O_\theta(z)
\|^2
+
\lambda_C
CE(
C_{\rm proj},
C_\theta(z)
)
\right]
}
$$

其中：

$$
(D_\theta,O_\theta,C_\theta)
=
DOAE_{\rm decoder}(z).
$$

得到：

$$
O^*
=
SurfaceCorrect(
O_\theta(z^*),M
).
$$

然后积分/生长：

$$
H_1
=
Trace(O^*).
$$

再：

$$
z_{\rm init}^{(2)}
=
Encoder(H_1)
$$

重新求：

$$
z_{\rm final}
=
\arg\min_zL_{\rm opt}(z).
$$

最后：

$$
\boxed{
H_{\rm final}
=
Trace(
SurfaceCorrect(
O_\theta(z_{\rm final}),M
))
}
$$

**这基本就是 HairLRM 全篇最核心的算法。**

而且从复现角度看，真正需要自己实现/拼起来的模块边界也已经非常清楚：**Hunyuan3D → canonical mesh fitting → SAM/HairStep → PyTorch3D rasterization + differential lifting → DOAE → latent inversion → surface tangent correction → CT2Hair/Newton growing → re-encode refinement**。论文明确指出 Hunyuan3D 只是实验所用 LRM，DOAE 只消费 mesh，因此前端 LRM 在架构上是可替换的。

需要注意一个复现边界：**论文给出了上述算法、公式和主要超参数，但没有把所有工程细节都写到能逐行复刻的程度**，尤其 CT2Hair 内部、forward Newton advection 的具体步长/终止条件、DOAE 每层 transformer 的精确宽度/层数等，在这篇论文正文和附录中没有完整列出。因此这些部分不能仅凭论文无损恢复源码。
