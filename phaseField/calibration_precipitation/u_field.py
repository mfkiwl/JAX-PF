"""
Implicit finite element solver for Eshelby inclusion problem
"""
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import os
import meshio
import sys
import glob
import scipy

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh, box_mesh
from jax_fem.utils import save_sol, modify_vtu_file, json_parse
from jax_fem.problem import Problem

from jax import config
config.update("jax_enable_x64", True)


onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)



class Inclusion(Problem):
    def custom_init(self, params):
        ## Hu: phase field variable
        self.fe_u = self.fes[0]
        self.params = params

        # isotropic elastic constants for Mg matrix
        E = self.params['E']
        nu = self.params['nu']
        
        # isotropic elastic constants for beta precipitate
        E_beta = self.params['E_beta']
        nu_beta = self.params['nu_beta']
        

        # A4, A3, A2, A1, and A0 Mg-Y matrix free energy parameters
        self.A4 = self.params['A4']
        self.A3 = self.params['A3']
        self.A2 = self.params['A2']
        self.A1 = self.params['A1']
        self.A0 = self.params['A0']

        # B2, B1, and B0 Mg-Y matrix free energy parameters
        self.B2 = self.params['B2']
        self.B1 = self.params['B1']
        self.B0 = self.params['B0']
        

        self.sfts_linear1 = np.array([[0., 0.],
                                      [0., 0.]])


        self.sfts_const1 = np.array([[0.0345, 0.],
                                     [0., 0.0185]])

        self.sfts_linear2 = np.array([[0., 0.],
                                      [0., 0.]])

        self.sfts_const2 = np.array([[0.0225, -0.0069],
                                     [-0.0069, 0.0305]])

        self.sfts_linear3 = np.array([[0., 0.],
                                      [0., 0.]])

        self.sfts_const3 = np.array([[0.0225, 0.0069],
                                     [0.0069, 0.0305]])


        E_gp = E*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))
        nu_gp = nu*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))
        E_beta_gp = E_beta*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))
        nu_beta_gp = nu_beta*onp.ones((len(self.fes[0].cells), self.fes[0].num_quads))

        self.internal_vars = [E_gp, nu_gp, E_beta_gp, nu_beta_gp]



    ## Hu: The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    ## Hu: solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    ## Hu: It has been distributed on each quad point
    def get_tensor_map(self):
        ### Hu: Interpolation function
        def h_local(n1, n2, n3):
            h1V = 10.0 * n1 * n1 * n1 - 15.0 * n1 * n1 * n1 * n1 + 6.0 * n1 * n1 * n1 * n1 * n1
            h2V = 10.0 * n2 * n2 * n2 - 15.0 * n2 * n2 * n2 * n2 + 6.0 * n2 * n2 * n2 * n2 * n2
            h3V = 10.0 * n3 * n3 * n3 - 15.0 * n3 * n3 * n3 * n3 + 6.0 * n3 * n3 * n3 * n3 * n3
            return h1V, h2V, h3V

        def elastic_modulus_cal(E, nu):
            C11 = E*(1-nu)/((1+nu)*(1-2*nu))
            C12 = E*nu/((1+nu)*(1-2*nu))
            C44 = E/(2.*(1. + nu))
            
            C = np.zeros((self.dim, self.dim, self.dim, self.dim))


            C = C.at[0, 0, 0, 0].set(C11)
            C = C.at[1, 1, 1, 1].set(C11)

            C = C.at[0, 0, 1, 1].set(C12)
            C = C.at[1, 1, 0, 0].set(C12)

            C = C.at[0, 1, 0, 1].set(C44)
            C = C.at[0, 1, 1, 0].set(C44)
            C = C.at[1, 0, 0, 1].set(C44)
            C = C.at[1, 0, 1, 0].set(C44)
            return C

        ## Hu: (u_grads, *internal_vars)
        def stress(u_grad, *internal_vars):
            sol_u_old, sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old, c_old, n1_old, n2_old, n3_old, \
            E, mu, E_beta, mu_beta = internal_vars

            h1V, h2V, h3V = h_local(n1_old, n2_old, n3_old)

            C_Mg = elastic_modulus_cal(E, mu)
            C_beta = elastic_modulus_cal(E_beta, mu_beta)
            CIJ_combined = C_Mg * (1.0 - h1V - h2V - h3V) + C_beta * (h1V + h2V + h3V)

            ## sfts = a_p * c_beta + b_p
            sfts1   = self.sfts_linear1 * c_old + self.sfts_const1
            sfts2   = self.sfts_linear2 * c_old + self.sfts_const2
            sfts3   = self.sfts_linear3 * c_old + self.sfts_const3

            epsilon = 0.5*(u_grad + u_grad.T)
            epsilon0 = sfts1*h1V + sfts2*h2V + sfts3*h3V

            E2 = epsilon - epsilon0

            sigma = np.sum(CIJ_combined[:, :, :, :] * E2[None, None, :, :], axis = (2, 3))
            
            return sigma
        return stress



    def set_initial_params(self, initial_params):
        # Override base class method.
        sol_u_old, sol_phaseField_list, quad_phaseField_old_list = initial_params

        sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list

        quad_c_old, quad_n1_old, quad_n2_old, quad_n3_old = quad_phaseField_old_list

        self.initial_internal_vars = [sol_u_old[self.fe_u.cells],
                              sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old,
                              quad_c_old, quad_n1_old, quad_n2_old, quad_n3_old]

        self.internal_vars = self.initial_internal_vars + self.internal_vars


    ## Hu: function for updating time-related parameters
    def update_internal_params(self, internal_params):
        sol_u_old, sol_phaseField_list, quad_phaseField_old_list = internal_params

        sol_c_old, sol_n1_old, sol_n2_old, sol_n3_old = sol_phaseField_list

        quad_c_old, quad_n1_old, quad_n2_old, quad_n3_old = quad_phaseField_old_list

        self.internal_vars[0] = sol_u_old[self.fe_u.cells]
        self.internal_vars[1] = sol_c_old
        self.internal_vars[2] = sol_n1_old
        self.internal_vars[3] = sol_n2_old
        self.internal_vars[4] = sol_n3_old
        self.internal_vars[5] = quad_c_old
        self.internal_vars[6] = quad_n1_old
        self.internal_vars[7] = quad_n2_old
        self.internal_vars[8] = quad_n3_old


    def set_initial_guess(self, initial_sol):
        self.initial_guess = initial_sol

    def set_params(self, params):
        # This is the key method for solving differentiable inverse problems.
        # We MUST define (override) 'set_params' method so that ``params`` become differentiable.
        # No need to define this method if only forward problem is solved.
        # See https://github.com/deepmodeling/jax-fem/blob/c3fbcb3ef9e44643a6afd59a20235a9def368c64/jax_fem/problem.py#L472
        self.internal_vars = params
