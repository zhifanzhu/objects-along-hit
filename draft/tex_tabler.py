## Latex table formatter

import pandas as pd


class TexTabler:
    def __init__(self, df):
        self.df = df
        self.best_prefix = '\\cellcolor{green!25}'
        self.second_prefix = '\\cellcolor{yellow!25}'

    @staticmethod
    def replace_underscore(s):
        return s.replace('_', '\_')

    def to_latex(self):
        df = self.df
        nrows, ncols = df.shape

        rows = []
        for i in range(nrows):
            for j, col in enumerate(df.columns):
                row = dict(
                    obj=df.iloc[i, 0],
                    metric=col[1],
                    method=col[0],
                    val=df.iloc[i, j],
                    ind=j
                )
                rows.append(row)
        frame = pd.DataFrame(rows)

        # obj_metric_rank = dict()
        # for obj in frame.obj.unique():
        #     for metric in frame.metric.unique():
        #         subframe = frame[(frame.obj == obj) & (frame.metric == metric)]
        #         # Save top 2 indices
        #         obj_metric_rank[(obj, metric)] = subframe.sort_values(
        #             'val', ascending=False).ind.values[:2].tolist()

        # Improved version of obj_metric_rank 
        # Break the tie, i.e. more than one green cell
        # method_to_rank[(obj, metric)][method] will give it's ranking 0(best), 1(second), 2(third)
        method_to_rank = dict()
        for obj in frame.obj.unique():
            for metric in frame.metric.unique():
                method_to_rank[(obj, metric)] = dict()
                subframe = frame[(frame.obj == obj) & (frame.metric == metric)]
                _sorted_vals = list(subframe.val.sort_values(ascending=False).values)
                for _j in subframe.index:
                    _val = subframe.val[_j]
                    _rank = _sorted_vals.index(_val)
                    _method_name = subframe.method[_j]
                    method_to_rank[(obj, metric)][_method_name] = _rank
        
        n_methods = len(df.columns.get_level_values(0)[1:].unique())
        n_metrics = len(df.columns.get_level_values(1)[1:].unique())
        text = "\\begin{tabular}{l" + ('|'+'c'*n_metrics)*n_methods + "}\n"
        text += "\\toprule\n"
        method_text = " & " + " & ".join(["\\multicolumn{%d}{c|}{%s}" % (n_metrics, self.replace_underscore(method) )for method in df.columns.get_level_values(0)[1:].unique()]) + " \\\\"
        metric_text = " Category & " + " & ".join(["%s" % metric for metric in df.columns.get_level_values(1)[1:]]) + " \\\\"
        text += method_text + "\n"
        text += metric_text + "\n"
        text += "\\midrule\n"

        # metric_te
        for i in range(nrows):
            # ss = "%s" % df.iloc[i, 0]
            ss = "%s" % df.iloc[i, 0].replace('_', '\_')  # latex underscore
            for j in range(1, ncols):
                val = float(df.iloc[i, j]) * 100
                obj_metric = (df.iloc[i, 0], df.columns[j][1])
                _method_name = df.columns[j][0]
                if method_to_rank[obj_metric][_method_name] == 0:  # Best / Co-Best
                    prefix = self.best_prefix
                elif method_to_rank[obj_metric][_method_name] == 1:  # Second / Co-Second
                    prefix = self.second_prefix
                else:
                    prefix = ""
                ss += " & " + prefix + "%.1f" % val
            ss += " \\\\"
            text += ss + "\n"
            
            if i +1 == nrows - 1:
                text += "\\hline\n"
        text += """\\bottomrule
\\end{tabular}"""
        return text