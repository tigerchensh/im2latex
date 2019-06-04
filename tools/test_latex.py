import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

ax = plt.axes()
ax.set_aspect('equal')
ax.set_axis_off()

prop = fm.FontProperties(fname='../fonts/CREDC.ttf')
txt = ax.text(0.2, 0.5,
              r'$\frac{\mathit{3}}{4} \binom{3}{4} \stackrel{3}{4}$',
              fontsize=25, fontproperties=prop)

# txt = ax.text(0.2, 0.5,
#               r'$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!',
#               fontsize=25, fontproperties=prop)

txt = ax.text(0.2, 0.8,
              '1.23',
              fontsize=25, fontproperties=prop)

# plt.xlabel(r'\textbf{time} (s)')
# plt.ylabel(r'\textit{voltage} (mV)', fontsize=16)
# plt.title(r"\TeX\ is Number "
#           r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#           fontsize=16, color='gray')
# Make room for the ridiculously large title.
# plt.subplots_adjust(top=0.8)

plt.savefig('tex_demo')
plt.show()
