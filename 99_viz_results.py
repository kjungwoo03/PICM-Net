import matplotlib.pyplot as plt
import numpy as np

uncompressed_baseline = {
    'resnet50_top1': 79.978,
    'convnext_top1': 83.682,
    'vit_large_21k_top1': 84.12,
    'clip_top1': 69.97,
}

results_dict = {
    'progdtd': {
        'bpp': [
            1.2040271965824827, 1.200774493266125, 1.2007452772588145, 1.200601793065363, 1.2006824588289067, 1.1559231986804885,
            1.0666479638644628, 0.9668690969749373, 0.8446714379349534, 0.7076808950122522, 0.6350484806664136, 0.5460864287249896,
            0.4459316210479152, 0.34367332136144446, 0.2085730658501995, 0.05651008219895314
        ],
        'mse': [
            0.0004966369831853793, 0.000496665453265079, 0.0004966759737058333, 0.0004965436267096321, 0.0004967174843804226, 0.0005219227002401437,
            0.000589196725834009, 0.0006888608896049994, 0.000850587760749254, 0.0011030751249600885, 0.0012811559931926277, 0.0015551602902670143,
            0.0019722299588540073, 0.0025934510631486773, 0.004098282502108842, 0.012157415296426234
        ],
        'psnr': [
            34.34217737158951, 34.34195327758789, 34.34216398122359, 34.343376081817006, 34.34192676933444, 34.16163646931551,
            33.704264348866985, 33.089726817851165, 32.225679553284934, 31.109692437308176, 30.451717026379644, 29.57723280848289,
            28.484138157902933, 27.21080022928666, 25.046585141396037, 19.93064411319032
        ],
        'ssim': [
            0.942135858900693, 0.9421427122184208, 0.9421407343173513, 0.9421426653862, 0.9421348912375314, 0.9399179141132199,
            0.9341365579439669, 0.9256812796300772, 0.9119526251238219, 0.8909330240317753, 0.8761213348836315, 0.8545839385110505,
            0.8229116882596698, 0.778530096521183, 0.6929730791218427, 0.5169216108565428
        ],
        'msssim': [
            0.9903628491625494, 0.9903633886454056, 0.9903629981741613, 0.9903628199684377, 0.9903618875814943, 0.9900802337393468,
            0.9892088588403196, 0.9877917651011019, 0.9851883643743943, 0.9806207077843803, 0.9772340929021641, 0.9713797283415891,
            0.9615892068463929, 0.9459235041725392, 0.9021461040389781, 0.7176128498145512
        ],
        'resnet50_top1': [
            75.834, 75.844, 75.844, 75.844, 75.844, 75.63, 75.29599999999999, 74.5, 72.978, 70.694, 68.718, 65.732, 61.256, 53.693999999999996, 35.23, 2.968
        ],
        'convnext_top1': [
            80.062, 80.078, 80.078, 80.078, 80.078, 79.884, 79.59, 78.798, 77.678, 75.68, 74.144, 71.502, 66.892, 60.01, 43.262, 5.42
        ],
        'vit_large_21k_top1': [
            82.796, 82.806, 82.806, 82.806, 82.806, 82.67, 82.35, 81.906, 81.202, 80.23599999999999, 79.646, 78.506, 76.39, 72.872, 60.726, 10.692
        ],
        'clip_top1': [
            67.296, 67.294, 67.294, 67.294, 67.294, 67.298, 66.986, 66.594, 65.938, 64.388, 63.32, 61.08, 57.422, 51.244, 34.202, 5.312
        ]
    }
}

# (2) plot (2x4 subplot)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

method_names = {"progdtd": "ProgDTD (CVPRW'23)"}
colors = {"progdtd": 'tab:blue'}

# 1행: mse, psnr, ssim, msssim
metric_names = ['mse', 'psnr', 'ssim', 'msssim']
ylabels = ['MSE (↓)', 'PSNR (↑)', 'SSIM (↑)', 'MS-SSIM (↑)']
for i, metric in enumerate(metric_names):
    for k in results_dict.keys():
        axes[0, i].plot(
            results_dict[k]['bpp'],
            results_dict[k][metric],
            marker='o',
            label=method_names.get(k, k.replace('_', ' ').title()),
            color=colors.get(k, None)
        )
    axes[0, i].set_xlabel('BPP')
    axes[0, i].set_ylabel(ylabels[i])
    axes[0, i].set_title(metric.upper())
    axes[0, i].grid(True)
    if i == 0:
        axes[0, i].legend()

# 2행: resnet50, convnext, vit, clip
# 2행: resnet50, convnext, vit, clip
net_metrics = ['resnet50_top1', 'convnext_top1', 'vit_large_21k_top1', 'clip_top1']
net_titles = [
    'ResNet50\nTop-1 (↑)',
    'ConvNeXt-Tiny\nTop-1 (↑)',
    'ViT-Large-21k\nTop-1 (↑)',
    'CLIP\nTop-1 (↑)'
]

for i, (metric, title) in enumerate(zip(net_metrics, net_titles)):
    ax = axes[1, i]

    for k in results_dict.keys():
        ax.plot(
            results_dict[k]['bpp'],
            results_dict[k][metric],
            marker='o',
            label=method_names.get(k, k.replace('_', ' ').title()),
            color=colors.get(k, None)
        )

    # 🔴 uncompressed baseline (horizontal dashed line)
    if metric in uncompressed_baseline:
        ax.axhline(
            y=uncompressed_baseline[metric],
            color='red',
            linestyle='--',
            linewidth=1.8,
            alpha=0.9,
            label='Uncompressed' if i == 0 else None  # legend 한 번만
        )

        # adjust y-limits so there's some space above the Uncompressed line
        # Find current limits and set upper limit to be a bit higher than baseline
        baseline = uncompressed_baseline[metric]
        ymin, ymax = ax.get_ylim()
        # target is 2~3% above baseline or whatever is needed to fit all lines
        upper_margin = max(1.05 * baseline, baseline + 2.0)
        new_ymax = max(ymax, upper_margin)
        ax.set_ylim(ymin, new_ymax)

    ax.set_xlabel('BPP')
    ax.set_ylabel('Top-1 (%)')
    ax.set_title(title)
    ax.grid(True)

    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('results/final.png', dpi=300, bbox_inches='tight')
plt.show()
