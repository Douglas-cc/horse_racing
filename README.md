# Profit Optimization Proposal in Horse Racing Betting: Exploratory, Prescriptive and Clustering Analysis

![Alternate Text](https://usagif.com/wp-content/uploads/gifs/gif-horse-92.gif)

# Objective

The main objective is to analyze the features and optimize the adjusted value (adj_value), resulting in higher profit using the Starting Price (SP). You can create new features, such as the difference between the initial price or my odds (my_odds) and the SP. Approach it to optimize adj_value and achieve positive profit in SP win and SP place.

It wouldn't require a trained model but, more importantly, provide an understanding of what features would be "good features" in a model. The main idea is to optimize net profit (PL) based on my odds (my_odds) that I modeled and the SP price for the race.

# Project Steps:

1. Exploratory Data Analysis (EDA)

2. Optimize my margin of error to the break-even point (anything above this price will be a value bet)

3. Develop additional logic (based on dataset attributes) to maximize profit at SP prices

4. Additional ideas related to profit and model optimization

# Struct Project

```
detected-instruments  # Project folder
├── data              # Local project data
├── src               # Exploratory Jupyter notebooks and scripts
├── pyproject.toml    # Identifies the project root
├── README.md         # README.md explaining your project
├── images            # save graphs 
```

# Installation Dependeces

```bash
curl -sSL https://pdm.fming.dev/install-pdm.py | python3 -
pdm install 
```
or install via file requirements.txt

```bash
pip install pip-tools
pip-compile
pip install -r requirements.txt
```

# Feature Descriptions

| Column              | Explanation                                                  |
| ------------------- | ------------------------------------------------------------ |
| race_id             | Unique identifier for the race                               |
| runner_id           | Unique identifier for the horse                              |
| meeting_date        | Date of the race                                             |
| course              | Meeting name of the race                                     |
| race_time           | Race time                                                    |
| race_timestamp      | Race date/time timestamp                                     |
| name                | Horse name                                                   |
| b_rank              | My model's ranking                                           |
| m_rank              | Market ranking                                               |
| rank_diff           | Ranking difference                                           |
| my_odds             | My model's odds                                              |
| early_price         | Betfair's early price                                        |
| value               | Value based on my model                                      |
| m_type              | Model version                                                |
| race_type           | Race type                                                    |
| bet_type            | Bet type                                                     |
| res_id              | Unique identifier for mapping                                |
| prize               | Prize money for the winner                                   |
| race_age            | Age range of the horses                                      |
| race_dist_in_meters | Race distance in meters                                      |
| race_going          | Track condition                                              |
| pos                 | Finishing position, if -1 means the horse lost               |
| draw                | Horse's starting position, not available for all types of races |
| lengths_behind      | Number of lengths behind the horse that finished in front     |
| uid                 | Unique identifier for mapping                                |
| jockey              | Jockey                                                       |
| trainer             | Trainer                                                      |
| horse_age           | Horse's age                                                  |
| horse_equip         | Equipment used by the horse                                  |
| wgh                 | Weight the horse is carrying                                 |
| off_rating          | Official rating assigned to this horse                        |
| isp                 | Industry Starting Price                                      |
| bsp_win             | Betfair Starting Price for win                               |
| bsp_place           | Betfair Starting Price for place                             |
| hi_lo               | Betfair In-Play prices with payout over GBP100                |
| runners             | Number of horses in the race                                 |
| ew_rules            | Betting rules each in decimal format (0, 0.25, 0.2)           |
| early_value_bw      | This is early_price divided by my_odds                        |
| adj_value           | This is a number I use as a margin of error                   |
| bw_win_proba        | This is 1 divided by my odds                                 |
| early_win_proba     | 1 divided by early_price                                     |
| early_value         | This is bw_win_proba minus early_win_proba                    |
| early_value_adj     | This is early_value minus adj_value (adj_value should be optimized) |
| stake               | Always set as 1                                              |
| res_win_early       | Equal to early_price minus 1 if the horse finished in first place, otherwise -1 |
| ew_places           | Number of places in this race (1,2,3,4)                       |
| res_place_early     | Equal to early_price minus 1 and then multiplied by ew_rules. All of this only if ew_places >= position, otherwise -1 |
| res_win_sp          | Result using the Starting Price (the analysis should focus on this price) |
| res_place_sp        | Placement result using the Starting Price                     |
| res_win_bf          | Result for win on Betfair                                    |
| res_place_bf        | Result for place on Betfair                                  |


# Optimization


## Defining the problem

1. First, there's an operation to remove rows with missing data from a DataFrame `df`. This is done with `df = df.dropna()`, which means that all rows with missing values are eliminated, leaving only rows with complete data.

2. Next, a class called `ProbHourseRacing` is defined, which inherits from the `ElementwiseProblem` class. This suggests that this class is used in some optimization context.

3. The constructor `__init__` of the `ProbHourseRacing` class takes several arguments, including `stake`, `my_odds`, `res_win_sp`, `res_place_sp`, `ew_rules`, and `horse_age`. These arguments seem to be related to horse racing betting information, such as the stake made (`stake`), the odds (`my_odds`), the results of bets on winners (`res_win_sp`), the results of bets on placers (`res_place_sp`), each-way betting rules (`ew_rules`), and the age of horses (`horse_age`).

4. Inside the constructor, some variables are initialized, including `self.n_var`, which seems to be the number of variables, and other optimization-related parameters such as the number of objectives (`n_obj`), the number of constraints (`n_constr`), and lower and upper bounds (`xl` and `xu`) for decision variables.

5. The `_evaluate` function is defined to calculate the objectives and constraints of the optimization problem. It appears that this optimization problem involves two objectives: maximizing the net profit on winning bets (`f1`) and minimizing the loss in the net value of placed bets (`f2`). The constraints `g1` and `g2` are defined to ensure that the age of horses is within a certain range and that the probability of winning is greater than 25.

6. Finally, there's an explanatory comment mentioning that the optimizer's constraints will attempt to achieve their objectives as much as possible but may not be completely achieved due to the nature of the data.

In summary, this code seems to define an optimization problem related to horse racing bets, where the goal is to maximize profit on winning bets and minimize losses on placed bets, subject to some constraints related to the age of horses and the probability of winning. This optimization problem can be solved using a specific optimization technique.


## Objective Functions

1. Function f1:

   The f1 function is defined as follows:

$$
f_1(x) = -1 * \sum_{i=1}^n (bet_i * (x - 1) * \cdot \text{opening\_win\_sp_i} - bet_i)
$$


   Where:
   - \(f1\) is the function you want to define.
   - \(n\) is the number of elements in \(stake\), \(res\_win\_sp\), and \(stake\).
   - \(stake_i\) is the ith element of \(stake\).
   - \(x\) is a variable.
   - \(res\_win\_sp_i\) is the ith element of \(res\_win\_sp\).

2. F2 Function:

   The f2 function is defined as follows:
   
$$
f_2(x) = \sum_{i=1}^n (bet_i * (1 - x * \cdot ew_rules_i) * \cdot res-place-sp_i + bet_i)
$$

   Where:

   - \(f2\) is the function you want to define.
   - \(n\) is the number of elements in \(stake\), \(ew\_rules\), and \(res\_place\_sp\).
   - \(stake_i\) is the ith element of \(stake\).
   - \(x\) is a variable.
   - \(ew\_rules_i\) is the ith element of \(ew\_rules\).
   - \(res\_place\_sp_i\) is the ith element of \(res\_place\_sp\).
   
## NSGA-II

**NSGA-II (Non-dominated Sorting Genetic Algorithm II)** is a multi-objective optimization algorithm based on genetic algorithms. It is designed to solve optimization problems that involve optimizing multiple objectives, i.e., situations where you want to find a set of solutions that are optimal with respect to various goals or criteria, and these criteria may be in conflict with each other.

Here are the key concepts and features of NSGA-II:

1. **Multi-objective Optimization**: NSGA-II is used when there is more than one objective to be optimized. For example, in engineering, you may want to optimize a design to minimize costs and maximize efficiency, and these two objectives may conflict.

2. **Non-dominated Sorting**: A fundamental feature of NSGA-II is the non-dominated sorting of solutions. This means that it not only seeks solutions that are good with respect to a single objective but also looks for solutions that are not dominated by any other solution in terms of all objectives.

3. **Selection Based on Pareto Front**: NSGA-II uses the idea of the "Pareto Front," which is a set of non-dominated solutions. It performs parent selection from this front, ensuring a diversity of solutions that are good in different aspects.

4. **Genetic Operators**: Like other genetic algorithms, NSGA-II uses operators such as crossover and mutation to generate new solutions from existing ones.

5. **Elitism**: NSGA-II incorporates elitism to ensure that the best solutions are always preserved in subsequent generations.

6. **Population**: The algorithm operates on a population of candidate solutions and evolves this population over several generations.

7. **Convergence**: NSGA-II is designed to converge toward the true Pareto Front, where it is not possible to improve one objective without worsening another.

8. **Configurable Parameters**: The algorithm has various configurable parameters, such as population size, crossover rate, mutation rate, among others, which can be adjusted according to the specific problem.

NSGA-II is widely used in various disciplines, including engineering, finance, urban planning, and product design, whenever there is a need to find optimal solutions with respect to multiple objectives that may be in conflict. It is a powerful tool for dealing with complex multi-objective optimization problems.


## Metrics Optimization

1. **cv_min (Minimum Convergence)**: Refers to the minimum convergence value of the Pareto front points found. It indicates how close the points are to approaching the ideal Pareto front.

2. **cv_avg (Average Convergence)**: Refers to the average convergence of the Pareto front points found. It indicates how well-distributed the points are along the Pareto front.

3. **n_gen (Number of Generations)**: The number of generations is the amount of iterations or generations executed by the optimization algorithm to find Pareto solutions.

4. **n_eval (Number of Function Evaluations)**: Indicates how many times the objective function was evaluated during optimization. It is an important indicator of computational efficiency.

5. **n_nds (Number of Non-Dominated Solutions)**: Represents the number of non-dominated solutions found. These solutions are the best solutions found so far and make up the Pareto front.

6. **eps (Objective Space Hypervolume)**: The hypervolume is an indicator that measures the quality of the approximation of the ideal Pareto front. The higher the hypervolume value, the better the quality of the approximation.

These indicators are used to assess the performance of multi-objective optimization algorithms and help researchers and engineers choose the most suitable algorithm for their specific problems. They provide information about convergence, distribution, and the quality of solutions found during multi-objective optimization.


## Front Pareto

![pareto](/images/front_pareto.png)

# Clustering in optimal solutions

Segmenting the profile of horses considered great for betting on jokei

![pareto](/images/clustering.png)
