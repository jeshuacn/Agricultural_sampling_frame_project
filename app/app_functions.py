import numpy as np
import pandas as pd
import random
import streamlit as st

# Simple Random Sampling
#@st.cache
@st.cache_data 
def random_sampling(df, sample_size):
    '''
    Returns a random sampled dataframe of length sample_size.
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: Number of rows to include in the sample.
    Returns:
        A sampled dataframe of length sample_size.
  '''   
    
    return df.sample(sample_size)

# Systematic Sampling
#@st.cache
@st.cache_data 
def systematic_sampling(df, sample_size):
    '''
    Returns a systematic sampled dataframe of length sample_size. 
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: used to calculate steps.

    Returns:
        A systematically sampled dataframe of length sample_size. 
    '''     

    # Population size
    N = df.shape[0]

    # Set step size / sampling interval
    sampling_interval = N/sample_size

    # Generate a list of indices for the sample
    sample_indices = [int(i * sampling_interval) for i in range(sample_size)]

  
    #df.iloc[::step]

    return df.iloc[sample_indices]

# Replicated Sampling
#@st.cache
@st.cache_data 
def replicated_sampling(df, sample_size):
  '''
    Returns a replicated sample of the input dataframe of length sample_size.
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: Number of rows to include in the sample.
    Returns:
        A replicated sample of the input dataframe with sample_size replicates. 
  '''  
  sample_size = sample_size // 2
  # Random sampling
  random = random_sampling(df,sample_size)

  # Systematic sampling
  systematic = systematic_sampling(df,sample_size)

  return pd.concat([random,systematic],axis = 0)

# Probability Proportional to Size Sampling
#@st.cache
@st.cache_data 
def pps_sampling(df, variable,sample_size):
    """Perform PPS sampling on a DataFrame.

      Args:
          df (dataFrame): The DataFrame to sample from.
          variable (str): The variable to base the sampling on. Can be either a numerical or categorical variable.
          sample_size (int): The size of the sample to return.

      Returns:
          A DataFrame containing the selected samples.
      """

    if df[variable].dtype == 'float' or df[variable].dtype == 'int':

        # Calculate the population size (total field area)
        population_size = df[variable].sum()

        # Calculate the sampling interval
        sampling_interval = population_size / sample_size

        # Calculate the probability of each crop being selected
        crop_probabilities = df[variable] / population_size
        
        # Calculate the cumulative sum of the probabilities
        #cum_sum = crop_probabilities.cumsum()
    else:

        # Calculate the population size (total number of rows)
        population_size = df[variable].count()

        # Calculate the sampling interval
        sampling_interval = population_size / sample_size

        # Calculate the probability of each crop being selected
        crop_counts = df[variable].value_counts()
        crop_probabilities = crop_counts / population_size

        # Calculate the cumulative sum of the probabilities
        #cum_sum = crop_probabilities.cumsum()

    # Select a random starting point
    start = df[variable].sample(1).iloc[0]

    # Select the samples
    samples = df[df[variable].eq(start)]

    # Select additional samples until the sample size is reached
    while len(samples) < sample_size:
      
        if variable in df.select_dtypes(include=['float', 'int']).columns:
            # Select the next crop based on its probability
            next_crop = random.choices(df.index, weights=crop_probabilities)[0]
            #next_crop = random.choices(df.index, weights=probabilities, cum_weights=cum_sum)[0]

        else:
            # Select the next crop based on its probability
            next_crop = random.choices(list(crop_counts.index), weights=list(crop_probabilities))[0]
            #next_crop = random.choices(list(crop_counts.index), weights=list(crop_probabilities), cum_weights=list(cum_sum))[0]
            

            
        samples = pd.concat([samples, df[df[variable].eq(next_crop)]])

    # Trim the sample to the desired size
    samples = samples.iloc[:sample_size]

    return samples


# Stratified Sampling
#@st.cache
@st.cache_data 
def stratified_sampling(df, sample_size):
    '''
    Returns a stratified sample of the input dataframe.
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: The number of rows to include in the sample.
        stratify_by: The column to stratify the sample by.
    
    Returns:
        A stratified sample of the input dataframe, with approximately equal numbers of rows in each stratum.
    '''

    field_area_median = df.Field_area.median()
    lower = df.loc[df.Field_area< field_area_median]
    upper = df.loc[df.Field_area>= field_area_median]

    lower_std = lower.Field_area.std()
    upper_std = upper.Field_area.std()

    lower_mean = lower.Field_area.mean()
    upper_mean = upper.Field_area.mean()

    lower_median = lower.Field_area.median()
    upper_median = upper.Field_area.median()

    C1 = lower_std/lower_mean
    C2 = upper_std/upper_mean

    lower_sample = round(sample_size*(C1*np.sqrt(lower_median))/(C1*np.sqrt(lower_median)+C2*np.sqrt(upper_median)))
    upper_sample = round(sample_size*(C2*np.sqrt(upper_median))/(C1*np.sqrt(lower_median)+C2*np.sqrt(upper_median)))

    lower_df = lower.sample(lower_sample)
    upper_df = upper.sample(upper_sample)

    strata_df = pd.concat([lower_df,upper_df],axis = 0)

    strata_df = strata_df.reset_index().drop(['ID'],axis = 1)

    return   strata_df, lower, upper, lower_sample, upper_sample, lower_df, upper_df


def show_values(axs, orient="v", df=[1], space=.01):
    '''
    Add values to bar plots.
    
    Parameters:
    axs (matplotlib.axes.Axes or numpy.ndarray): The axes object(s) to add values to.
    orient (str, optional): The orientation of the bars. Can be "v" (vertical) or "h" (horizontal). Default is "v".
    space (float, optional): The space between the bar and the value label. Default is 0.01.
    '''
    # Helper function to add values to a single axes object
    def _single(ax):
        if orient == "v":

          # Iterate through the patches (individual bars)
            for p in ax.patches:

               # Calculate the x and y position of the value label
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)

                # Format the value as a string and add it to the plot
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
                
        elif orient == "h":
            for p in ax.patches:

                # Calculate the x and y position of the value label
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)

                # Format the value as a string and add it to the plot
                value = '{:.1f}%'.format(p.get_width()/len(df)*100)
                ax.text(_x, _y, value, ha="left")
                
    # If axs is an array, apply the function to each element
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)

    # If axs is a single axes object, apply the function to it
    else:
        _single(axs)  


mk = '''The Dataset contains 6 variables, among which 5 are categorical (qualitative) and 1 is numerical (quantitative):

1. **Production mode:**
    
    * **Primary:** The same crop is planted year after year
    * **In succession:** A different crop is planted each year (or each cycle)
    * **In association:** Multiple crops are planted on the same field at the same time
    * **Understory:** Crop is planted under trees (Can be in a forest)

2. **Irrigation:** 
    * **Yes:** The farmer uses a water source besides rain
    * **No:** The farmer is strictly relying on rain (Pluvial or Rainfed Irrigation mode)
3. **Irrigation mode:**

    * **Localized:** Farmer uses a drip irrigation system (continuous drops of water). This method is the most efficient in terms of water usage.
    * **Gravity:** An open air canal linking the field to the water source (eg. river). This irrigation system uses a lot of water, especially since much of it is lost through land absorption on the way and evaporation.
    * **Aspersion:** Water is brought to the plants in the form of artificial rain using sprinklers fixed across the field.
    * **Pivot:** A mobile system that pumps water from a source to a long tube with sprinklers in the sky that crosses the entire field. This sky tube moves from one side to the other and irrigates the whole field.
    * **Gravity, Localized:** Mixed
    * **Localized, Pivot:** Mixed

4. **Crop:**
    * Tomatoes
    * Potatoes
    * ...
5. **Greenhouse:**
    * **No:** Not used
    * **Small tunel:** A tunnel that's small in width but high in length
    * **Big Tunel:** A tunnel that's high in width but small in length
    * **Canarian:** A structure made of only wires and films
    * **Multi-chapel:** Greenhouse with more than one chapel (curve)

6. **Field area:**
    * Planted field area in Ha (Hectares)'''