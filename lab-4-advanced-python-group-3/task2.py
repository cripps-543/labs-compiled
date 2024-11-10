import random


class RandomNumberGenerator():
    
    def __init__(self, seed, lower_bound, upper_bound):
        random.seed(seed)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def random_int(self):
        # Your code goes here
        return random.randint(self.lower_bound, self.upper_bound)
    
    def random_float(self):
        # Your code goes here
        return random.uniform(self.lower_bound, self.upper_bound)
    
    
if __name__ == '__main__':
    random_number_generator = RandomNumberGenerator(seed=10, lower_bound=0, upper_bound=100)
    print(random_number_generator.random_int())
    print(random_number_generator.random_float())
    