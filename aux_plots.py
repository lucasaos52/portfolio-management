from datetime import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

def ploty_basic(df,x_data,y_data,mode_plot='line',title=None,
                y_title=None,x_title=None,type_plot="lines",
                multi_yaxes=False,anotations=None,width = 900,height = 700,
               not_pair = True,color_background='white',showgrid=True):
        

    if not isinstance(y_data, list):
        y_data = [y_data]
        x_data = [x_data]
        names = y_data.copy()
        
    else:
        if not isinstance(x_data, list):
            x_data = len(y_data)*[x_data]
            names = y_data.copy()
        else:
            x_data = len(y_data)*x_data
            names = y_data.copy()
      
    
    # todos os tracos iguaos
    if not isinstance(type_plot, list):
        if len(y_data)!=1:
            type_plot = len(y_data)*[type_plot]
        
        else:
            type_plot = [type_plot]
    
    else:
        
        if len(y_data)==1:
            type_plot = len(y_data)*type_plot
        

    '''
    
     - Caso se deseje destacar os pontos no grafico. Recebe o dataframe de pontos que se deseja 
    destacar
    
     - Esta implementado apenas para 1 plot
    
    '''
    
    lista_dict = []
    
    
    # anotacoes simples
    if ((anotations is not None)& (not_pair)):
        
        # dicionario layout
        d1 = dict(x=4,y=4,xref='x',yref='y',text='Annotation Text 2',showarrow=True,arrowhead=7,ax=0,ay=-40)
        
        lista_dict = []
        
        vals_x = anotations[x_data[0]].tolist()
        vals_y = anotations[y_data[0]].tolist()
        for el in range(len(vals_y)):
            dd = d1.copy()
            dd["x"] = vals_x[el]
            dd["y"] = vals_y[el]
            dd["text"] = 'trades_{}'.format(el)
            lista_dict.append(dd)
       
    
    # anotacoes de trades contendo o par de compra e venda de cada trade
    elif ((anotations is not None)& (not not_pair)):
        
        lista_dict = []
        
        vals_x_buy = anotations[x_data[0]+'_buy'].tolist()
        vals_y_buy = anotations[y_data[0]+'_buy'].tolist()
        vals_x_sell = anotations[x_data[0]+'_sell'].tolist()
        vals_y_sell = anotations[y_data[0]+'_sell'].tolist()

        d1 = dict(x=4,y=4,xref='x',yref='y',text='Annotation Text 2',showarrow=True,arrowhead=7,ax=0,ay=-40,arrowcolor='#636363')
        
        for el in range(len(vals_y_sell)):
            dd = d1.copy()
            dd2 = d1.copy()
            dd["x"] = vals_x_buy[el]
            dd["y"] = vals_y_buy[el]
            dd["text"] = 'trades_buy_{}'.format(el)
            dd["arrowcolor"] = '#636363'
            lista_dict.append(dd)

            dd2["x"] = vals_x_sell[el]
            dd2["y"] = vals_y_sell[el]
            dd2["text"] = 'trades_sell_{}'.format(el)
            dd2["arrowcolor"] = '#d9f441'
            lista_dict.append(dd2)
      
    
    data = []
    
    
    ## criamos uma lista de traces
    
    count = 1
        
    for el in range(len(x_data)):
        
        if mode_plot == 'line':
            trace = go.Scatter(
                        x=df['{}'.format(x_data[el])],
                        y=df['{}'.format(y_data[el])],
                        name = names[el],
                       mode = type_plot[el],
                       yaxis='y{}'.format(count)
                      )
                
        elif mode_plot =='bar':
            trace = go.Bar(
                x=df['{}'.format(x_data[el])],
                y=df['{}'.format(y_data[el])],
                name = names[el],
                opacity = 0.8)

        else:
            print("tipo invalido de Modo de plot")
        
        if multi_yaxes:
            count +=1
        

        data.append(trace)
    

    
    if not multi_yaxes:
    
        layout = dict(
            width=width,
            height=height,
            title = '{}'.format(title),
            yaxis = dict(title='{}'.format(y_title),showgrid=True,gridcolor='#bdbdbd'),
            xaxis = dict(title='{}'.format(x_title),showgrid=True,gridcolor='#bdbdbd'),
            annotations=lista_dict,
            #showgrid = showgrid,
            plot_bgcolor=color_background
            )
    
    else:
        
        layout = go.Layout(
            width=width,
            height=height,
            title=title,
            yaxis=dict(
                title='yaxis title',
                showgrid=True,gridcolor='#bdbdbd'
            ),
            yaxis2=dict(
                showgrid=True,gridcolor='#bdbdbd',
                title='yaxis2 title',
                titlefont=dict(
                    color='rgb(148, 103, 189)'
                ),
                tickfont=dict(
                    color='rgb(148, 103, 189)'
                ),
                overlaying='y',
                side='right'
            ),
        annotations=lista_dict,
        plot_bgcolor=color_background
        )
       
        
    #ata = [trace]

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename = "-")
    
    return 0


'''
 
 - benchmarkl is the equal weights

'''
def plot_results(benchmark_series, 
                 target_series, 
                 target_balances, 
                 n_assets,
                 columns,
                 name2plot = '',
                 path2save = './',
                 base_name_series = 'series'):
    
    import matplotlib
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='red')
    #N = len(np.array(benchmark_series).cumsum())
    N = len(np.array([item for sublist in benchmark_series for item in sublist]).cumsum()) 
    
    if not os.path.exists(path2save):
        os.makedirs(path2save)

    for i in range(0, len(target_balances)):
        
        current_range = np.arange(0, N)
        current_ts = np.zeros(N)
        current_ts2 = np.zeros(N)

        ts_benchmark = np.array([item for sublist in benchmark_series[:i+1] for item in sublist]).cumsum()
        ts_target = np.array([item for sublist in target_series[:i+1] for item in sublist]).cumsum()

        t = len(ts_benchmark)
        current_ts[:t] = ts_benchmark
        current_ts2[:t] = ts_target

        current_ts[current_ts == 0] = ts_benchmark[-1]
        current_ts2[current_ts2 == 0] = ts_target[-1]

        plt.figure(figsize = (12, 10))
        
        plt.subplot(2, 1, 1)
        plt.bar(np.arange(n_assets), target_balances[i], color = 'grey')
        plt.xticks(np.arange(n_assets), columns, rotation='vertical')

        plt.subplot(2, 1, 2)
        plt.colormaps = current_cmap
        plt.plot(current_range[:t], current_ts[:t], color = 'black', label = 'Benchmark')
        plt.plot(current_range[:t], current_ts2[:t], color = 'red', label = name2plot)
        plt.plot(current_range[t:], current_ts[t:], ls = '--', lw = .1, color = 'black')
        plt.autoscale(False)
        plt.ylim([-1.5, 1.5])
        plt.legend()
        plt.savefig(path2save + base_name_series + str(i) + '.jpg')