# Source: https://github.com/axelfahy/bff
from collections import abc
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pandas.api.types import is_hashable
import seaborn as sns


def cast_to_category_pd(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """
    Automatically converts columns of pandas DataFrame that are worth stored as ``category`` dtype.

    To be casted a column must not be numerical, must be hashable and must have less than 50%
    of unique values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the columns to cast.
    deep : bool, default True
        Whether to perform a deep copy of the original DataFrame.

    Returns
    -------
    pd.DataFrame
        Optimized copy of the input DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> columns = ['name', 'age', 'country']
    >>> df = pd.DataFrame([['John', 24, 'China'],
    ...                    ['Mary', 20, 'China'],
    ...                    ['Jane', 25, 'Switzerland'],
    ...                    ['Greg', 23, 'China'],
    ...                    ['James', 28, 'China']],
    ...                   columns=columns)
    >>> df
        name  age      country
    0   John   24        China
    1   Jane   25  Switzerland
    2  James   28        China
    >>> df.dtypes
    name       object
    age         int64
    country    object
    dtype: object
    >>> df_optimized = cast_to_category_pd(df)
    >>> df_optimized.dtypes
    name       object
    age         int64
    country  category
    dtype: object
    """
    return (df.copy(deep=deep)
            .astype({col: 'category' for col in df.columns
                     if (df[col].dtype == 'object'
                         and is_hashable(df[col].iloc[0])
                         and df[col].nunique() / df[col].shape[0] < 0.5)
                     }
                    )
            )


def plot_confusion_matrix(y_true: Union[np.array, pd.Series, Sequence],
                          y_pred: Union[np.array, pd.Series, Sequence],
                          labels_filter: Optional[Union[np.array, Sequence]] = None,
                          ticklabels: Any = 'auto',
                          sample_weight: Optional[str] = None,
                          normalize: Optional[str] = None,
                          stats: Optional[str] = None,
                          annotation_fmt: Optional[str] = None,
                          cbar_fmt: Optional[FuncFormatter] = None,
                          title: str = 'Confusion matrix',
                          ax: Optional[plt.axes] = None,
                          rotation_xticks: Union[float, None] = 90,
                          rotation_yticks: Optional[float] = None,
                          figsize: Tuple[int, int] = (13, 10),
                          dpi: int = 80,
                          style: str = 'white') -> plt.axes:
    """
    Plot the confusion matrix.

    The confusion matrix is computed in the function.

    Parameters
    ----------
    y_true : np.array, pd.Series or Sequence
        Actual values.
    y_pred : np.array, pd.Series or Sequence
        Predicted values by the model.
    labels_filter : array-like of shape (n_classes,), default None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    ticklabels : 'auto', bool, list-like, or int, default 'auto'
        If True, plot the column names of the DataFrame. If False, don’t plot the column names.
        If list-like, plot these alternate labels as the xticklabels. If an integer,
        use the column names but plot only every n label. If “auto”,
        try to densely plot non-overlapping labels.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    normalize : str {'true', 'pred', 'all'}, optional
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    stats : str {'accuracy', 'precision', 'recall', 'f1-score'}, optional
        Calculate and display the wanted statistic below the figure.
    annotation_fmt : str, optional
        Format for the annotation on the confusion matrix.
        If not provided, default value is ',d' or '.2f' if normalize is given.
    cbar_fmt : FuncFormatter, optional
        Formatter for the colorbar. Default is with thousand separator and one decimal.
        If normalize and not provided, cbar format is not changed.
    title : str, default 'Confusion matrix'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    rotation_xticks : float or None, default 90
        Rotation of x ticks if any.
    rotation_yticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    figsize : Tuple[int, int], default (13, 10)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'white'
        Style to use for seaborn.axes_style.
        The style is use only in this context and not applied globally.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> y_true = ['dog', 'cat', 'bird', 'cat', 'dog', 'dog']
    >>> y_pred = ['cat', 'cat', 'bird', 'dog', 'bird', 'dog']
    >>> plot_confusion_matrix(y_true, y_pred, stats='accuracy')
    """
    from sklearn.metrics import classification_report, confusion_matrix  # pylint: disable=C0415

    # Compute the confusion matrix.
    conf_matrix = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                                   labels=labels_filter, normalize=normalize)

    with sns.axes_style(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        if ticklabels in (True, 'auto') and labels_filter is None:
            ticklabels = sorted(set(list(y_true) + list(y_pred)))

        if annotation_fmt is None:
            annotation_fmt = '.2g' if normalize else ',d'

        if cbar_fmt is None:
            if np.max(conf_matrix) > 1_000:
                cbar_fmt = FuncFormatter(lambda x, p: format(int(x), ',d'))
        # Draw the heatmap with the mask and correct aspect ratio.
        # pylint: disable=E1101
        sns.heatmap(conf_matrix, cmap=plt.cm.Blues, ax=ax, annot=True, fmt=annotation_fmt,
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.75, 'format': cbar_fmt},
                    xticklabels=ticklabels, yticklabels=ticklabels)

        if stats:
            report = classification_report(y_true, y_pred,
                                           labels=labels_filter,
                                           target_names=ticklabels,
                                           sample_weight=sample_weight,
                                           output_dict=True)
            if stats == 'accuracy':
                ax.text(1.05, 0.05, f'{report[stats]:.2f}', horizontalalignment='left',
                        verticalalignment='center', transform=ax.transAxes)
            else:
                # Depending on the metric, there is one value by class.
                # For each class, print the value of the metric.
                for i, label in enumerate(ticklabels):
                    if stats in report[label].keys():
                        ax.text(1.05, 0.05 - (0.03 * i),
                                f'{label}: {report[label][stats]:.2f}',
                                horizontalalignment='left',
                                verticalalignment='center', transform=ax.transAxes)
                    else:
                        LOGGER.error(f'Wrong key {stats}, possible values: '
                                     f'{list(report[label].keys())}.')
            # Print the metric used.
            if stats in report.keys() or stats in report[ticklabels[0]].keys():
                ax.text(1.05, 0.08, f'{stats.capitalize()}', fontweight='bold',
                        horizontalalignment='left',
                        verticalalignment='center', transform=ax.transAxes)

        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(title, fontsize=14)
        # Style of the ticks.
        plt.xticks(fontsize=12, alpha=1, rotation=rotation_xticks)
        plt.yticks(fontsize=12, alpha=1, rotation=rotation_yticks)

        return ax


def format_axis(ax: plt.axes) -> plt.axes:
    """
    Format the given axis.

    - Remove border on top and right.
    - Set alpha on the remaining borders.
    - Only show ticks on the left and bottom spines.
    - Change the style of the ticks (alpha).

    Parameters
    ----------
    ax : plt.axes
        Axes from matplotlib on which to apply the formatting.

    Returns
    -------
    plt.axes
        The formatted axis.
    """
    # Style.
    # Remove border on the top and right.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set alpha on remaining borders.
    ax.spines['left'].set_alpha(0.4)
    ax.spines['bottom'].set_alpha(0.4)

    # Only show ticks on the left and bottom spines.
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Style of ticks.
    plt.xticks(fontsize=10, alpha=0.7)
    plt.yticks(fontsize=10, alpha=0.7)
    return ax


def get_n_colors(n: int, cmap: str = 'rainbow') -> List:
    """
    Get `n` colors from a color map.

    A color is represented by an array having 4 components (r, g, b, a).
    A list of array is return containing the `n` colors.

    Parameters
    ----------
    n : int
        Number of colors to get.
    cmap : str, default 'rainbow'
        Color map for the colors to retrieve.

    Returns
    -------
    list
        List of colors from the color map.
    """
    assert cmap in plt.colormaps(), f'Colormap {cmap} does not exist.'
    return list(cm.get_cmap(cmap)(np.linspace(0, 1, n)))


def normalization_pd(df: pd.DataFrame, scaler=None,
                     columns: Optional[Union[str, Sequence[str]]] = None,
                     suffix: Optional[str] = None, new_type: np.dtype = np.float32,
                     **kwargs) -> pd.DataFrame:
    """
    Normalize columns of a pandas DataFrame using the given scaler.

    If the columns are not provided, will normalize all the numerical columns.

    If the original columns are integers (`RangeIndex`), it is not possible to replace
    them. This will create new columns having the same integer, but as a string name.

    By default, if the suffix is not provided, columns are overridden.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    scaler : TransformerMixin, default MinMaxScaler
        Scaler of sklearn to use for the normalization.
    columns : sequence of str, default None
        Columns to normalize. If None, normalize all numerical columns.
    suffix : str, default None
        If provided, create the normalization in new columns having this suffix.
    new_type : np.dtype, default np.float32
        New type for the columns.
    **kwargs
        Additional keyword arguments to be passed to the
        scaler function from sklearn.

    Returns
    -------
    pd.DataFrame
        DataFrame with the normalized columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = {'x': [123, 27, 38, 45, 67], 'y': [456, 45.4, 32, 34, 90]}
    >>> df = pd.DataFrame(data)
    >>> df
         x      y
    0  123  456.0
    1   27   45.4
    2   38   32.0
    3   45   34.0
    4   67   90.0
    >>> df_std = df.pipe(normalization_pd, columns=['x'], scaler=StandardScaler)
    >>> df_std
              x      y
    0  1.847198  456.0
    1 -0.967580   45.4
    2 -0.645053   32.0
    3 -0.439809   34.0
    4  0.205244   90.0
    >>> df_min_max = normalization_pd(df, suffix='_norm', feature_range=(0, 2),
    ...                               new_type=np.float64)
    >>> df_min_max
         x      y    x_norm    y_norm
    0  123  456.0  2.000000  2.000000
    1   27   45.4  0.000000  0.063208
    2   38   32.0  0.229167  0.000000
    3   45   34.0  0.375000  0.009434
    4   67   90.0  0.833333  0.273585
    """
    if scaler is None:
        _check_sklearn_support('normalization_pd')
        from sklearn.preprocessing import MinMaxScaler  # pylint: disable=C0415
        scaler = MinMaxScaler
    # If columns are not provided, select all the numerical columns of the DataFrame.
    # If provided, select only the numerical ones.
    cols_to_norm = ([col for col in value_2_list(columns) if np.issubdtype(df[col], np.number)]
                    if columns else df.select_dtypes(include=[np.number]).columns)
    return df.assign(**{f'{col}{suffix}' if suffix else str(col):
                        scaler(**kwargs).fit_transform(df[[col]].values.astype(new_type))
                        for col in cols_to_norm})


def set_thousands_separator(axes: plt.axes, which: str = 'both',
                            nb_decimals: int = 1) -> plt.axes:
    """
    Set thousands separator on the axes.

    Parameters
    ----------
    axes : plt.axes
        Axes from matplotlib, can be a single ax or an array of axes.
    which : str, default 'both'
        Which axis to format with the thousand separator ('both', 'x', 'y').
    nb_decimals: int, default 1
        Number of decimals to use for the number.

    Returns
    -------
    plt.axes
        Axis with the formatted axes.

    Examples
    --------
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(...)
    >>> set_thousands_separator(ax, which='x', nb_decimals=3)
    """
    for ax in np.asarray(value_2_list(axes)).flatten():
        # Set a thousand separator for axis.
        if which in ('x', 'both'):
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: f'{x:,.{nb_decimals}f}')
            )
        if which in ('y', 'both'):
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: f'{x:,.{nb_decimals}f}')
            )
    return axes


def value_2_list(value: Any) -> Sequence:
    """
    Convert a single value into a list with a single value.

    If the value is alredy a sequence, it is returned without modification.
    Type `np.ndarray` is not put inside another sequence.

    Strings are not considered as a sequence in this scenario.

    Parameters
    ----------
    value
        Value to convert to a sequence.

    Returns
    -------
    sequence
        Value put into a sequence.

    Examples
    --------
    >>> value_2_list(42)
    [42]
    >>> value_2_list('Swiss')
    ['Swiss']
    >>> value_2_list(['Swiss'])
    ['Swiss']
    """
    if (not isinstance(value, (abc.Sequence, np.ndarray)) or isinstance(value, str)):
        value = [value]
    return value
