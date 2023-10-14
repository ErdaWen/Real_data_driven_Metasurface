import matplotlib.pyplot as plt
import numpy as np

def setup_figure(xlabel,ylabel,
                 fontsize = 15,
                 fontname = "Helvetica",
                 grid = False,has_legend = True,
                 **kwargs):
    plt.xlabel(xlabel,fontsize = fontsize,fontname = fontname)
    plt.xticks(fontsize=fontsize,fontname = fontname)
    plt.ylabel(ylabel,fontsize = fontsize,fontname = fontname)
    plt.yticks(fontsize=fontsize,fontname = fontname)
    if grid: 
        plt.grid(which = 'major')
        if grid == "minor":
            plt.minorticks_on()
            plt.grid(which = 'minor')
    if "xlim" in kwargs: plt.xlim(kwargs["xlim"])
    if "ylim" in kwargs: plt.ylim(kwargs["ylim"])
    if "title" in kwargs: 
        plt.title(kwargs["title"],fontsize = fontsize,fontname = fontname)
    if has_legend: plt.legend()
    
def setup_polar(fig,ax,ylabel,ylim,
                mrate,ydiv = 4,fontsize = 15,
                fontname = "Helvetica",**kwargs):
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    # ax.set_thetagrids([],
    #                   fontsize=fontsize,fontname = fontname)
    ax.set_thetagrids([0,30,60,90,120,150,180],
                      fontsize=fontsize,fontname = fontname)
    ax.set_ylim(ylim)
    ax.set_rorigin(ylim[0]-(ylim[1]-ylim[0])*mrate)
    fig.canvas.draw()
    ax.set_rgrids(np.linspace(*ylim,ydiv),
                  fontsize=fontsize,fontname = fontname)

    ax.xaxis.grid(visible=False)
    if "title" in kwargs: ax.set_title(kwargs["title"],fontsize = fontsize,fontname = fontname)
    ax.legend()
    
def sector_plot(fig,ax,x,y,ylim,
                crate,mrate,label = None,
                ydiv = 4,fontsize = 15,
                fontname = "Helvetica",fmt = None,stem = False,show_dir = False,alpha = 1):
    if stem:
        if fmt:
            ax.stem(np.deg2rad(90+(x-90)*crate),y,fmt[0],fmt[1],label = label,basefmt=" ")
            #ax.stem(np.deg2rad(90+(x-90)*crate),y,fmt[0],fmt[1],label = label,basefmt=" ",bottom = ylim[1])
        else:
            ax.stem(np.deg2rad(90+(x-90)*crate),y,label = label,basefmt=" ")
            #ax.stem(np.deg2rad(90+(x-90)*crate),y,label = label,basefmt=" ",bottom = ylim[1])
    else:
        if fmt:
            ax.plot(np.deg2rad(90+(x-90)*crate),y,fmt,label = label,alpha = alpha)
        else:
            ax.plot(np.deg2rad(90+(x-90)*crate),y,label = label,alpha = alpha)
        if show_dir:
            for i in range(x.size):
                if not np.isnan(y[i,0]):
                    ax.stem(np.deg2rad(90+(x[i]-90)*crate),1000,linefmt = 'k--')
    ax.set_thetamin(90-90*crate)
    ax.set_thetamax(90+90*crate)
    thetagrids = np.array([0,30,60,90,120,150,180])
    thetagridslabel = ["0°","30°","60°","90°","120°","150°","180°"]
    ax.set_thetagrids(90+(thetagrids-90)*crate,
                      labels = thetagridslabel,
                      fontsize=fontsize,fontname = fontname)
    ax.set_ylim(ylim)
    ax.set_rorigin(ylim[0]-(ylim[1]-ylim[0])*mrate)
    fig.canvas.draw()
    ax.set_rgrids(np.linspace(*ylim,ydiv),
                  fontsize=fontsize,fontname = fontname)
    ax.xaxis.grid(visible=False)
    if label is not None:
        ax.legend()
        
def sector_label(fig,ax,xlabel,ylabel,ylim,crate,tickbot = True,xlabelpos = 0.33,ylabelpos= 1.55):
    if tickbot: ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.tick_right()
    if xlabel is not None:
        ax.text(np.pi/2,ylim[0]-(ylim[1]-ylim[0])*xlabelpos,xlabel,ha='center',va='center')
    rot = 90+90*crate
    if rot>145: rot -= 180
    ax.text(np.pi/2+np.pi/2*crate*ylabelpos,(ylim[1]+ylim[0])/2,ylabel,
            rotation = rot,ha='center',va='center')
    fig.canvas.draw()