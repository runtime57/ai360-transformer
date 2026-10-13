from pathlib import Path
import math
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, Polygon, String, PolyLine, Group
from reportlab.graphics import renderPDF, renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color, white
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent
FONTS = Path('/Users/rokatasonov/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
pdfmetrics.registerFont(TTFont('DV', str(FONTS / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVB', str(FONTS / 'DejaVuSans-Bold.ttf')))
W, H = 1080, 710
INK = HexColor('#152C40')
MUTED = HexColor('#52687A')
BLUE = HexColor('#087FAD')
PURPLE = HexColor('#7150AF')
LIGHT = HexColor('#EDF5F8')
GRAY = HexColor('#D5E1E9')

def text(d, x, y, s, size=24, color=INK, bold=False, anchor='start'):
    d.add(String(x, H-y, s, fontName='DVB' if bold else 'DV', fontSize=size, fillColor=color, textAnchor=anchor))

def rect(d, x, y, w, h, fill=white, stroke=None, radius=12):
    d.add(Rect(x, H-y-h, w, h, rx=radius, ry=radius, fillColor=fill, strokeColor=stroke, strokeWidth=1.5))

def line(d, x1, y1, x2, y2, color=GRAY, width=2):
    d.add(Line(x1,H-y1,x2,H-y2,strokeColor=color,strokeWidth=width))

def arrow(d, x1,y1,x2,y2,color=INK,width=2.6):
    line(d,x1,y1,x2,y2,color,width)
    a=math.atan2(y2-y1,x2-x1)
    points=[x2,H-y2,x2-11*math.cos(a-.43),H-(y2-11*math.sin(a-.43)),x2-11*math.cos(a+.43),H-(y2-11*math.sin(a+.43))]
    d.add(Polygon(points,fillColor=color,strokeColor=None))

def box(d,x,y,w,h,label,color=INK,fill=white,size=24):
    rect(d,x,y,w,h,fill,color)
    text(d,x+w/2,y+h/2+size*.34,label,size,color,True,'middle')

def base(title, subtitle, color):
    d=Drawing(W,H)
    rect(d,0,0,W,H,white,None,0)
    rect(d,0,0,W,9,color,None,0)
    text(d,42,72,title,44,color,True)
    text(d,42,111,subtitle,24)
    return d

def rotation(d,cx,cy,angle,label):
    r=78
    d.add(Circle(cx,H-cy,r,fillColor=None,strokeColor=GRAY,strokeWidth=2))
    line(d,cx-r-12,cy,cx+r+12,cy,GRAY,1.5)
    line(d,cx,cy-r-12,cx,cy+r+12,GRAY,1.5)
    arrow(d,cx,cy,cx+r,cy,MUTED,2)
    arrow(d,cx,cy,cx+r*math.cos(angle),cy-r*math.sin(angle),BLUE,4)
    if angle:
        pts=[]
        for k in range(31):
            a=angle*k/30
            pts.extend([cx+37*math.cos(a),H-(cy-37*math.sin(a))])
        d.add(PolyLine(pts,strokeColor=BLUE,strokeWidth=2,fillColor=None))
    text(d,cx,cy+r+24,label,25,INK,True,'middle')

rope=base('RoPE','Позиция токена задаёт угол поворота Q и K.',BLUE)
text(rope,42,162,'Поворот каждой пары координат',24,INK,True)
rect(rope,42,182,996,272,LIGHT)
rotation(rope,210,307,0,'m = 0')
rotation(rope,540,307,.55,'m = 1')
rotation(rope,870,307,1.1,'m = 2')
arrow(rope,321,307,423,307,BLUE)
arrow(rope,651,307,753,307,BLUE)
text(rope,373,283,'+θᵢ',25,BLUE,True,'middle')
text(rope,703,283,'+θᵢ',25,BLUE,True,'middle')
text(rope,540,442,'Одинаковый вектор показан в трёх позициях.',18,MUTED,False,'middle')
text(rope,42,492,'q′ₘ = Rₘ qₘ     k′ₙ = Rₙ kₙ',30,INK,True)
text(rope,42,532,'Пары координат (2i, 2i + 1) поворачиваются на угол mθᵢ.',23)
text(rope,42,588,'θᵢ = 10000',30,INK,True)
exponent_x = 45 + pdfmetrics.stringWidth('θᵢ = 10000', 'DVB', 30)
text(rope,exponent_x,570,'−2i/dₕ',24,INK,True)
text(rope,363,588,'•  V без поворота  •  0 обучаемых параметров',23,MUTED)
rect(rope,42,609,996,65,HexColor('#DDF0F6'))
text(rope,63,651,'(q′ₘ)ᵀk′ₙ = qₘᵀ Rₙ₋ₘ kₙ',28,BLUE,True)
text(rope,593,650,'Позиционный вклад зависит от n − m.',21,INK)
text(rope,42,699,'Su et al. · RoFormer · arXiv:2104.09864',18,MUTED)

gate=base('Gated attention','Обучаемый sigmoid-gate регулирует выход attention.',PURPLE)
text(gate,42,162,'Gate после SDPA, перед выходной проекцией',24,INK,True)
rect(gate,42,181,996,224,HexColor('#F4F0FA'))
box(gate,64,283,90,62,'x',INK,white,27)
box(gate,198,283,137,62,'Q, K, V',INK,white,24)
box(gate,379,283,138,62,'SDPA',INK,white,24)
box(gate,569,204,196,58,'Linear → σ',PURPLE,white,22)
gate.add(Circle(667,H-314,29,fillColor=white,strokeColor=PURPLE,strokeWidth=2))
text(gate,667,324,'⊙',34,PURPLE,False,'middle')
box(gate,773,283,126,62,'Wₒ',INK,white,28)
text(gate,976,324,'y',29,INK,True,'middle')
arrow(gate,154,314,198,314)
arrow(gate,335,314,379,314)
arrow(gate,517,314,638,314)
text(gate,577,297,'A',22,MUTED,False,'middle')
arrow(gate,696,314,773,314,PURPLE)
arrow(gate,899,314,954,314)
line(gate,109,283,109,233,PURPLE,2.6)
arrow(gate,109,233,569,233,PURPLE)
arrow(gate,667,262,667,285,PURPLE)
text(gate,690,282,'g',21,PURPLE,True)
text(gate,540,385,'g = σ(Linear(x))     y = Wₒ Concat(Aₕ ⊙ gₕ)',25,INK,False,'middle')
text(gate,42,448,'Headwise',26,PURPLE,True)
text(gate,564,448,'Elementwise',26,PURPLE,True)
text(gate,42,481,'Один gate на голову и токен.',23)
text(gate,564,481,'Один gate на координату и токен.',23)
values1=[[.85]*4,[.18]*4,[.55]*4]
values2=[[.85,.18,.5,.92],[.12,.75,.3,.55],[.6,.25,.9,.15]]
for offset,vals in [(104,values1),(626,values2)]:
    for row,vs in enumerate(vals):
        text(gate,offset-17,522+row*36,'h'+str(row+1),18,MUTED,False,'end')
        for col,v in enumerate(vs):
            shade=Color(1-(1-PURPLE.red)*v,1-(1-PURPLE.green)*v,1-(1-PURPLE.blue)*v)
            rect(gate,offset+col*55,499+row*36,47,28,shade,None,4)
    if offset == 104:
        text(gate,offset+106,634,'gₕ ∈ (0, 1)',24,INK,False,'middle')
    else:
        label = 'gₕ ∈ (0, 1)'
        label_width = pdfmetrics.stringWidth(label, 'DV', 24)
        d_width = pdfmetrics.stringWidth('d', 'DV', 18)
        h_width = pdfmetrics.stringWidth('h', 'DV', 13)
        label_x = offset + 106 - (label_width + d_width + h_width) / 2
        text(gate,label_x,634,label,24)
        text(gate,label_x+label_width,620,'d',18)
        text(gate,label_x+label_width+d_width,625,'h',13)
text(gate,42,670,'g ≈ 0: подавить выход    •    g ≈ 1: сохранить выход',24,INK,True)
text(gate,42,699,'Qiu et al. · Gated Attention · arXiv:2505.06708',18,MUTED)

scale=380*72/25.4/W
for name,d in [('rope',rope),('gated-attention',gate)]:
    renderSVG.drawToFile(d,str(OUT/(name+'.svg')))
    c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(380*72/25.4,250*72/25.4))
    c.setTitle('Poster method: '+name)
    c.scale(scale,scale)
    renderPDF.draw(d,c,0,0)
    c.save()
c=canvas.Canvas(str(OUT/'methods-pair.pdf'),pagesize=(780*72/25.4,250*72/25.4))
c.setTitle('RoPE and gated attention: A0 poster modules')
c.scale(scale,scale)
renderPDF.draw(rope,c,0,0)
renderPDF.draw(gate,c,W+20*72/25.4/scale,0)
c.save()
print('Created two 380 × 250 mm modules and 780 × 250 mm paired PDF.')
