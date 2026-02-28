# from scvx import SCvxProblem
# from models.quadcopter_fixedft import QuadcopterFixedFT
import cvxpy as cvx
import numpy as np

def main():
    # model = QuadcopterFixedFT()
    # problem = SCvxProblem(model)
    # problem.solve(verbose=True)
    x = np.array([2, 3, 4])
    y = np.array([2, 3, 4])
    # print(cvx.vdot(x, y).value)
    print(cvx.flatten(y[:, None]))


if __name__ == "__main__":
    main()