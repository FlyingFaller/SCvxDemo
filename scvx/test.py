from scvx import SCvxProblem
from models.quadcopter_fixedft import QuadcopterFixedFT

def main():
    model = QuadcopterFixedFT()
    problem = SCvxProblem(model)
    problem.solve(verbose=True)


if __name__ == "__main__":
    main()