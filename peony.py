"""Процедурная анимация махрового пиона. Запуск: python peony.py"""
import argparse
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio_ffmpeg
import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFilter


WIDTH, HEIGHT, FPS, DURATION = 800, 1000, 30, 19
TAU = 2 * math.pi
ROWS, COLS = 34, 55
T, V = np.meshgrid(np.linspace(0, 1, ROWS), np.linspace(-1, 1, COLS), indexing="ij")


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def project(r, w, z, angle, ox=0., oy=0.):
    x = ox + r*np.cos(angle) - w*np.sin(angle)
    y = oy + r*np.sin(angle) + w*np.cos(angle)
    return np.stack((x, -.58*y-.815*z, -.815*y+.58*z), axis=-1)


def cubic(a, b, c, d, u):
    return (1-u)**3*a + 3*(1-u)**2*u*b + 3*(1-u)*u*u*c + u**3*d


def flower(seed, bud=False):
    rng = np.random.default_rng(seed)
    closed, unfurl, opened, motion, materials = [], [], [], [], []
    # Радиус, полуширина, высота кромки. Верхние лепестки загибаются
    # обратно к оси, образуя плотную шапку внутри наружной чаши.
    rings = [(8,1.02,.55,.58), (10,.91,.46,.72), (12,.80,.40,.82),
             (13,.68,.345,.91), (14,.54,.28,.99), (12,.43,.23,1.03)]
    arrangement = []
    for layer, (number, radius, width, height) in enumerate(rings[:3] if bud else rings):
        offset = rng.uniform(0,TAU)
        for j in range(number):
            angle = offset+(j+rng.uniform(-.17,.17))*TAU/number
            arrangement.append((layer,angle,radius,width,height,0.,0.))
    if not bud:
        # Несколько неравных групп коротких внутренних петалодиев.
        for cx,cy,number,radius,h in [(-.16,.12,12,.22,1.10),(.13,.14,10,.21,1.07),
                                      (.11,-.13,12,.22,1.12),(-.13,-.10,9,.18,1.06)]:
            for j in range(number):
                f = (j+.5)/number
                angle = j*2.399963 + 9*cx
                arrangement.append((6+f,angle,radius*(1.2-.75*f),.15-.053*f,
                                    h+.09*f,cx,cy))
    for layer, angle, radius, width, height, ox, oy in arrangement:
        p,q,s = rng.uniform(0,TAU,3)
        inner = min(1.,layer/5)
        radius *= rng.uniform(.90,1.11)
        width *= rng.uniform(.85,1.20)
        height += rng.uniform(-.075,.07)
        tilt = rng.uniform(-.12,.12)
        wave = .55*np.sin((3.8+2.7*inner)*V+p+.46*T) + .30*np.sin(10*V+q-1.3*T)
        fine = .38*np.sin(18*V+s+1.7*T) + .19*np.sin(29*V+p-1.2*T)
        # Скруглённые плечи и отдельные мелкие выемки на краю лепестка.
        shoulder = rng.uniform(.23,.38)
        notch = rng.uniform(.012,.044)*np.exp(-((V-rng.uniform(-.5,.5))/.14)**2)
        u = T*(1-T**2*(shoulder*(1-np.cos(np.pi*V/2)) + notch))
        pleat = (wave+.37*fine)*u**1.6
        root = max(.018,.14-.017*layer)
        curl = rng.uniform(.05,.14) + .12*inner
        r = cubic(root, radius*.59, radius*(1.06+curl), radius*(1-curl),u)
        z = cubic(.025*layer, -.025+.03*layer, height*(.78+.15*inner), height,u)
        # У наружных лепестков мягкая отогнутая кромка, у внутренних — складка.
        lip = rng.uniform(.09,.23)*(1-.64*inner)
        r += lip*u**5
        z -= lip*1.10*u**5
        cup = (1-np.cos(1.48*V)) * np.sin(np.pi*u*.62)**1.2
        r -= width*(.57+.12*inner)*cup
        z += width*.26*cup + (tilt*.58+.080)*V*u
        r += (.038+.078*inner)*pleat
        z += (.037+.068*inner)*pleat + (.019+.018*inner)*fine*u**4
        z += (.024+.015*inner)*np.sin(8*V+q+.58*np.sin(6*V+p))*u**3
        roll = (.05+.09*inner)*u**3*np.cos(3.1*V+p)
        r += roll
        silk = np.sin((34+4*np.cos(p))*V + 2*np.sin(4*V+q) + 1.4*u)
        r += .0037*silk*u**1.7
        z -= (.06+.07*inner)*u**4 + .50*roll
        w = width*(.012+.988*np.sin(np.pi*.65*u)**1.13)*np.sin(1.52*V)
        w += rng.uniform(-.17,.17)*u*u + (.035+.027*inner)*pleat
        angle_open = angle + (.18+rng.uniform(-.23,.23))*u + .22*inner*V*u
        opened.append(project(r,w,z,angle_open,ox*u,oy*u))
        # Бутон — вложенные округлые оболочки с широким нахлёстом.
        envelope = 1 - .067*layer
        rc = envelope*(.07*(1-u)+.315*np.sin(np.pi*u*.995)**.74+.010*u)
        zc = .018*layer+(.66-.027*layer)*u + .008*pleat
        wrap = (.82+.34*np.sin(np.pi*u))*V
        rc += .014*np.sin(np.pi*u)*np.sin(3*V+q+u)
        zc += .018*u*u*np.cos(p+2.6*V)
        closed.append(project(rc,0.,zc,angle+wrap+.30*u))
        rm = cubic(root,.40,.58-.025*layer,.33-.024*layer,u)
        zm = cubic(.018*layer,.025,.55,.92+.008*layer,u)
        unfurl.append(project(rm-.12*cup + .022*pleat,
                              w*.66,zm+.02*pleat+.07*V**2*u,angle+.10*u,ox*u*.4,oy*u*.4))
        start = 2.7 + .82*layer + rng.uniform(-.58,.63)
        motion.append((start,rng.uniform(4.8,6.3),rng.uniform(.65,1.30),p))
        materials.append((p,q,rng.uniform(.92,1.08),0.,rng.uniform(-.055,.055),inner))
    for j in range(5):
        angle = TAU*j/5+.42
        width = .20*(.006+np.sin(np.pi*T)**.78)*V
        # Чашелистики обнимают нижнюю половину закрытого бутона.
        ts = T*(.66+.06*np.cos(angle))
        rs = .076*(1-ts)+.353*np.sin(np.pi*ts)**.76
        a = project(rs-.042*V**2*np.sin(np.pi*T),
                    width,.66*ts-.025*V**2*np.sin(np.pi*T),angle)
        b = project(.09+.38*T,width,.17*np.sin(2.4*T)-.055*T,angle)
        closed.append(a)
        unfurl.append(.55*a+.45*b)
        opened.append(b)
        motion.append((2.4+.21*j,4.4,.65,j))
        materials.append((j,j*.6,.96,1.,0.,0.))
    return tuple(np.ascontiguousarray(x,dtype=np.float32)
                 for x in (closed,unfurl,opened,motion,materials))


@njit
def deform(closed, control, opened, motion, sec, limit):
    mesh = np.empty_like(closed)
    for k in range(len(closed)):
        start, duration, lag, phase = motion[k]
        for i in range(closed.shape[1]):
            t = i/(closed.shape[1]-1)
            for j in range(closed.shape[2]):
                v = 2*j/(closed.shape[2]-1)-1
                time = (sec-start-lag*(.76*t+.25*v*t))/duration
                time = min(1.,max(0.,time))
                u = limit*time**3*(10+time*(-15+6*time))
                # Дифференциальное раскрытие и лёгкая отдача тонкого края.
                bow = math.sin(math.pi*u)*t*t
                recoil = .017*math.sin(6.6*u+phase)*bow
                breath = .0036*t*t*u*math.sin(.91*sec+phase+1.2*v)
                for a in range(3):
                    mesh[k,i,j,a] = ((1-u)**2*closed[k,i,j,a]
                                     + 2*(1-u)*u*control[k,i,j,a] + u*u*opened[k,i,j,a])
                mesh[k,i,j,0] += (.023*bow+breath)*math.cos(phase)
                mesh[k,i,j,1] += recoil+breath
    return mesh


# Ортонормированный базис площадного источника в координатах камеры.
LIGHT = np.array([-.53,-.67,.52])
LIGHT /= np.linalg.norm(LIGHT)
LX = np.cross(LIGHT,[0.,1.,0.])
LX /= np.linalg.norm(LX)
LY = np.cross(LIGHT,LX)
LIGHT_BASIS = np.ascontiguousarray(np.stack((LX,LY,LIGHT)).T,dtype=np.float32)
SHADOW_SIZE = 416
SHADOW_SCALE = 149.


@njit
def shadow_map(light_points):
    depth = np.full((SHADOW_SIZE,SHADOW_SIZE),-10.,np.float32)
    for surface in light_points:
        for i in range(surface.shape[0]-1):
            for j in range(surface.shape[1]-1):
                for half in range(2):
                    ax,ay,az = surface[i+half,j+half]
                    bx,by,bz = surface[i+1,j]
                    cx,cy,cz = surface[i,j+1]
                    ux,uy,vx,vy = bx-ax,by-ay,cx-ax,cy-ay
                    det = ux*vy-uy*vx
                    if abs(det)<1e-6:
                        continue
                    for y in range(max(0,math.ceil(min(ay,by,cy))),min(SHADOW_SIZE-1,math.floor(max(ay,by,cy)))+1):
                        for x in range(max(0,math.ceil(min(ax,bx,cx))),min(SHADOW_SIZE-1,math.floor(max(ax,bx,cx)))+1):
                            b = ((x-ax)*vy-(y-ay)*vx)/det
                            c = (ux*(y-ay)-uy*(x-ax))/det
                            if b>=0 and c>=0 and b+c<=1:
                                depth[y,x] = max(depth[y,x],az+b*(bz-az)+c*(cz-az))
    return depth


@njit
def shade_mesh(points, light_points, shadow, prop):
    rows,cols = points.shape[:2]
    shades = np.empty((rows,cols),np.float32)
    taps = ((0,0),(-3,-1),(3,1),(-1,3),(1,-3),(5,-3),(-5,3),(-3,-5),(3,5))
    for i in range(rows):
        t = i/(rows-1)
        for j in range(cols):
            i0,i1,j0,j1 = max(0,i-1),min(rows-1,i+1),max(0,j-1),min(cols-1,j+1)
            ax = points[i1,j,0]-points[i0,j,0]
            ay = points[i1,j,1]-points[i0,j,1]
            az = points[i1,j,2]-points[i0,j,2]
            bx = points[i,j1,0]-points[i,j0,0]
            by = points[i,j1,1]-points[i,j0,1]
            bz = points[i,j1,2]-points[i,j0,2]
            nx,ny,nz = ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx
            norm = math.sqrt(nx*nx+ny*ny+nz*nz)+1e-10
            if nz<0:
                norm = -norm
            nx,ny,nz = nx/norm,ny/norm,nz/norm
            incidence = nx*LIGHT[0] + ny*LIGHT[1] + nz*LIGHT[2]
            px,py,pz = light_points[i,j]
            visibility = 0.
            for dx,dy in taps:
                x = min(SHADOW_SIZE-1,max(0,round(px)+dx))
                y = min(SHADOW_SIZE-1,max(0,round(py)+dy))
                gap = shadow[y,x]-pz
                visibility += min(1.,max(0.,1.14-4.8*gap))
            visibility /= len(taps)
            # Тонкая ткань пропускает тёплый свет даже на теневой стороне.
            diffuse = .25 + .29*max(0.,incidence) + .10*abs(incidence)
            fill = -.048*points[i,j,0]-.045*points[i,j,1]
            shades[i,j] = prop[2]*(.075 + .22*t + (.38+.62*visibility)*diffuse
                                   + .047*(1-nz)**2*visibility -.14*(1-visibility)+fill)
    return shades


@njit
def paint_triangles(points, shades, prop, depth, near, under):
    rows, cols = shades.shape
    h, w = depth.shape
    for i in range(rows-1):
        for j in range(cols-1):
            for half in range(2):
                ai, aj = i+half, j+half
                ax, ay, az = points[ai, aj]
                bx, by, bz = points[i+1, j]
                cx, cy, cz = points[i, j+1]
                dx1, dy1, dx2, dy2 = bx-ax, by-ay, cx-ax, cy-ay
                det = dx1*dy2-dy1*dx2
                if abs(det) < 1e-6:
                    continue
                x0 = max(0, math.ceil(min(ax, bx, cx)))
                x1 = min(w-1, math.floor(max(ax, bx, cx)))
                y0 = max(0, math.ceil(min(ay, by, cy)))
                y1 = min(h-1, math.floor(max(ay, by, cy)))
                for y in range(y0, y1+1):
                    for x in range(x0, x1+1):
                        b = ((x-ax)*dy2 - (y-ay)*dx2)/det
                        c = (dx1*(y-ay) - dy1*(x-ax))/det
                        a = 1-b-c
                        if min(a,b,c) < -1e-6:
                            continue
                        z = a*az + b*bz + c*cz
                        # Два ближайших слоя дают просвечивание без зависимости
                        # от порядка обхода лепестков.
                        if z <= under[y,x,0] or abs(z-depth[y,x]) < .0001:
                            continue
                        t = (a*ai+b*(i+1)+c*i)/(rows-1)
                        v = 2*(a*aj+b*j+c*(j+1))/(cols-1)-1
                        # Экранная ширина кромки остаётся тонкой даже в центре.
                        tu = ((i+1-ai)*dy2-(i-ai)*dy1)/(det*(rows-1))
                        tv = (dx1*(i-ai)-dx2*(i+1-ai))/(det*(rows-1))
                        vu = 2*((j-aj)*dy2-(j+1-aj)*dy1)/(det*(cols-1))
                        vv = 2*(dx1*(j+1-aj)-dx2*(j-aj))/(det*(cols-1))
                        dist = min((1-t)/(math.hypot(tu,tv)+1e-7),
                                   (1-abs(v))/(math.hypot(vu,vv)+1e-7))
                        rim = math.exp(-max(0.,dist)/1.20)*(.23+.77*t)
                        body = a*shades[ai,aj]+b*shades[i+1,j]+c*shades[i,j+1]
                        phase = 45*v+3.1*math.sin(4*v+prop[0])+.7*t+prop[1]
                        fibre = (.5+.5*math.cos(phase+1.7*math.sin(t*2.3+v)))**15
                        if prop[3] == 0:
                            dye = prop[4] + .023*math.cos(prop[0]+5*v+1.4*t)
                            branching = math.cos(89*v+3.8*math.sin(5*v+prop[0])+2.2*t+2.6*math.sin(phase*.43+t))
                            fine = .009*branching + .007*math.sin(133*v+.55*phase+18*t)
                            value = max(.015,min(1.09,body+dye+fine*t+.029*fibre*t+.085*rim))
                            # Розовая ткань: тёплый полутон и перламутровая кромка.
                            red = 37 + 213*(1-(1-min(value,1.))**2.05)
                            green = 9 + 222*value**1.34
                            blue = 32 + 193*value**1.15
                            green += 9*prop[5]*t
                            blue += 6*t
                        else:
                            stripe = .55*math.exp(-abs(v)*20)+.16*fibre
                            red = 6 + 53*body + 14*stripe + 8*rim
                            green = 15 + 86*body + 19*stripe + 16*rim
                            blue = 16 + 42*body + 11*stripe + 9*rim
                        if z > depth[y,x]:
                            under[y,x,0] = depth[y,x]
                            for k in range(3):
                                under[y,x,k+1] = near[y,x,k]
                            depth[y,x] = z
                            near[y,x,0], near[y,x,1], near[y,x,2] = red, green, blue
                        else:
                            under[y,x,0] = z
                            under[y,x,1], under[y,x,2], under[y,x,3] = red, green, blue


def draw_head(model, sec, scale, x, y, turn, limit, depth, near, under):
    closed, control, opened, motion, props = model
    mesh = deform(closed,control,opened,motion,sec,limit)
    light_points = mesh @ LIGHT_BASIS
    light_points[..., :2] *= SHADOW_SCALE
    light_points[..., :2] += SHADOW_SIZE/2
    shadow = shadow_map(light_points)
    co,si = math.cos(turn),math.sin(turn)
    for points,illum,prop in zip(mesh,light_points,props):
        shades = shade_mesh(points,illum,shadow,prop)
        px = points[:,:,0].copy()
        py = points[:,:,1].copy()
        points[:,:,0] = x+scale*(co*px-si*py)
        points[:,:,1] = y+scale*(si*px+co*py)
        points[:,:,2] *= scale
        paint_triangles(points,shades,prop,depth,near,under)


def bezier(points, t):
    a,b,c,d = np.array(points)
    t = np.asarray(t)[..., None]
    return (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d


def stalk(draw, curve, growth, width):
    if growth <= 0:
        return
    path = bezier(curve, np.linspace(0, growth, 100))
    # Узкая полоска света идёт по одной стороне стебля.
    for dx, size, rgb in [(0,width+2,(12,28,25)), (0,width,(25,52,36)),
                          (-1,max(1,width//2),(43,76,45)), (-1.5,1,(68,98,64))]:
        draw.line([(round(x+dx),round(y)) for x,y in path], fill=rgb, width=size)


def leaflet(image, base, tip, arc, width, growth):
    if growth < .015:
        return
    base, tip = np.asarray(base), np.asarray(tip)
    axis = (tip-base)*growth
    length = np.linalg.norm(axis)
    tangent = axis/length
    normal = np.array([-tangent[1],tangent[0]])
    end = base+axis
    margin = (width+abs(arc))*growth+2
    x0,y0 = np.maximum(0,np.floor(np.minimum(base,end)-margin).astype(int))
    x1,y1 = np.minimum(image.size,np.ceil(np.maximum(base,end)+margin).astype(int))
    if x1<=x0 or y1<=y0:
        return
    yy,xx = np.mgrid[y0:y1,x0:x1]
    px,py = xx-base[0],yy-base[1]
    t = np.clip((px*tangent[0]+py*tangent[1])/length,0,1)
    across = px*normal[0]+py*normal[1] - growth*arc*np.sin(np.pi*t)
    margin = width*growth*np.sin(np.pi*t)**.91*(.69+.55*t)
    margin *= 1-.24*np.exp(-((t-.60)/.065)**2) - .13*np.exp(-((t-.76)/.046)**2)
    v = np.clip(across/(margin+.001),-1,1)
    fold = .37+.31*np.sin(2.7*v+.45)+.15*t+.08*np.cos(5*t+2*v)
    midrib = np.exp(-abs(across)/.76)*.63
    vein_phase = 7.4*t-.95*abs(v)**.83+.11*np.sin(4.7*t)
    veins = np.exp(-(np.sin(np.pi*vein_phase)/.088)**2)*(1-abs(v))*.22
    light = fold + midrib + veins
    rgb = np.stack((9+37*light,20+50*light,23+29*light),axis=-1).astype(np.uint8)
    mask = 255*np.clip(margin-abs(across)+.5,0,1)*(t>0)*(t<1)
    image.paste(Image.fromarray(rgb),(int(x0),int(y0)),Image.fromarray(mask.astype(np.uint8)))


class Scene:
    def __init__(self):
        self.main = flower(94)
        self.bud = flower(16, bud=True)
        self.dust = np.random.default_rng(413).random((21,6))
        yy, xx = np.mgrid[:HEIGHT,:WIDTH]
        haze = np.exp(-((xx-375)/310)**2-((yy-386)/240)**2)
        self.back = np.uint8(np.stack((4+8*haze, 5+3*haze, 8+5*haze),axis=-1))

    def frame(self, sec):
        growth = ease((sec+.30)/3.95)
        weight = ease((sec-4.4)/9.1)
        sway = 2.3*math.sin(.82*sec+.5)+.8*math.sin(1.24*sec)
        nod = .018*math.sin(sec*.75)+.105*weight
        main_xy = (393-24*weight+sway, 475+16*weight+.35*sway)
        target_bud = (594+.9*sway,634+.40*sway)
        image = Image.fromarray(self.back)
        draw = ImageDraw.Draw(image)
        trunk = [(338,1040),(350,844),(452+.52*sway,606),main_xy]
        tip = bezier(trunk,growth)
        stalk(draw,trunk,growth,6)
        branch_base = bezier(trunk,.43)
        branch_growth = ease((sec-1.46)/3.55)
        sprig = [branch_base,(487,744),(555+1.1*sway,721),target_bud]
        bud_tip = bezier(sprig,branch_growth)
        stalk(draw,sprig,branch_growth,4)
        leaf_base = bezier(trunk,.29)
        leaves = ease((sec-1.22)/3.8)
        joint = leaf_base + np.array([-50.,-32.])*leaves
        stalk(draw,[leaf_base,leaf_base+(-16,-7),joint+(11,7),joint],leaves,3)
        for end,arc,width in [((142,721),-14,22),((216,694),9,16),((177,822),8,17)]:
            leaflet(image,joint,(end[0]+1.1*sway,end[1]+.5*sway),arc,width,leaves)
        for u,v,s,phase,rate,drift in self.dust:
            fade = max(0., math.sin(sec*.59 + TAU*phase))**4 * ease(sec/3)
            x = 48 + 700*u + 8*math.sin(sec*.32+drift*TAU)
            y = 130 + 680*v - sec*(.7+1.4*rate)
            if fade > .03:
                radius = .35 + .85*s
                col = tuple(int(a+fade*b) for a,b in [(5,101),(4,70),(8,77)])
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=col)
        base = np.asarray(image).astype(np.float32)
        depth = np.full((HEIGHT,WIDTH),-1e5,np.float32)
        near = np.zeros_like(base)
        under = np.zeros((HEIGHT,WIDTH,4),np.float32)
        under[:,:,0] = -1e5
        for model, growth, size, pos, angle, limit in [
                (self.main,.12+.88*ease((sec+.23)/4.2),285,tip,-.23+nod+.48*(1-ease((sec-.45)/4.7)),1.),
                (self.bud,ease((sec-1.46)/3.95),140,bud_tip,.36+.30*(1-branch_growth)+.016*math.sin(.73*sec-.8),.06)]:
            if growth > .01:
                draw_head(model,sec,size*growth,*pos,angle,limit,depth,near,under)
        mask = depth > -1e5
        beneath = np.where(under[:,:,0:1] > -1e5, under[:,:,1:], base)
        base[mask] = .94*near[mask] + .06*beneath[mask]
        image = Image.fromarray(np.uint8(np.clip(base,0,255)))
        # Ореол строится только из ярких частей уже нарисованного кадра.
        light = image.resize((WIDTH//4,HEIGHT//4),Image.Resampling.BOX)
        light = light.point([max(0,int((i-16)*.86)) for i in range(256)]*3)
        soft = light.filter(ImageFilter.GaussianBlur(2.1)).resize(image.size,Image.Resampling.BILINEAR)
        halo = light.filter(ImageFilter.GaussianBlur(7.0)).resize(image.size,Image.Resampling.BILINEAR)
        return np.uint8(np.clip(base + .085*np.asarray(soft) + .095*np.asarray(halo),0,255))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o","--output",type=Path,default=Path("peony.mp4"))
    path = parser.parse_args().output
    scene = Scene()
    # Готовый ролик остаётся доступен до окончания следующего рендера.
    with NamedTemporaryFile(dir=path.parent,suffix=".mp4",delete=False) as temporary:
        pending = Path(temporary.name)
    writer = imageio_ffmpeg.write_frames(str(pending), (WIDTH,HEIGHT), fps=FPS,
                codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
                output_params=["-crf","16","-preset","fast","-movflags","+faststart"],
                quality=None)
    try:
        writer.send(None)
        try:
            for index in range(FPS*DURATION):
                writer.send(scene.frame(index/FPS))
                if index % (FPS*2) == 0:
                    print(f"{index//FPS} / {DURATION} s",flush=True)
        finally:
            writer.close()
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    print(path.resolve())


if __name__ == "__main__":
    main()
