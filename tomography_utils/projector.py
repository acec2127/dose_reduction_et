"""
Compute projection matrix.
"""

import numpy as np
import numba as nb


@nb.njit(cache=True, parallel=False)
def _fill_projector_row(matrix,
                        row_ind,
                        theta, shift, radius,
                        tol=1e-15,
                        verbose=False,
                        warnings=False
                        ):
    
    c = np.cos(theta)
    s = np.sin(theta)
    R = radius
    sh = shift    
    if (sh > R*np.sqrt(2)): # line does not intersect the bulk
        return 0.0

    # get image size, pixel size
    npixels = int(np.sqrt(matrix.shape[1]))
    dx = 2*R / npixels # pixel's side length 
    direction = np.array([-s, c]) # geometry of the line
    
    # 1. find intersections of the line with borders [-radius,radius]x[-radius, radius]
    nintersect_pts = 0
    line_end_pts = np.zeros((2,2))
    
    # 1.a bottom border
    if (abs(c) > tol and abs(sh + R*s) < abs(c)*R):
        line_end_pts[nintersect_pts, 0] = (sh + R*s)/c
        line_end_pts[nintersect_pts, 1] = -R
        nintersect_pts += 1
    # 1.b right border
    if (abs(s) > tol and np.abs(sh - R*c) < np.abs(s)*R):
        line_end_pts[nintersect_pts, 0] = R
        line_end_pts[nintersect_pts, 1] = (sh - R*c)/s
        nintersect_pts += 1 
    # 1.c top border
    if (abs(c) > tol and np.abs(sh - R*s) < np.abs(c)*R):
        line_end_pts[nintersect_pts, 0] = (sh - R*s)/c
        line_end_pts[nintersect_pts, 1] = R
        nintersect_pts += 1 
            
    # 1.d left border
    if (abs(s) > tol and np.abs(sh + R*c) < np.abs(s)*R):
        line_end_pts[nintersect_pts, 0] = -R
        line_end_pts[nintersect_pts, 1] = (sh + R*c)/s
        nintersect_pts += 1

    if (nintersect_pts != 2):
        if (warnings):
            if (verbose):
                print('Precision Warning: for phi=', theta, ' shift=', shift, '; ',
                    'Potentially the line is too close to the domain boundary.')
            else:
                print('Precision Warning: some lines were close to the boundary - integral recieved 0.0')
        return 0.
    
    # if end-points are not aligned with direction - swap
    if (np.dot(line_end_pts[1]-line_end_pts[0], direction) < 0.0):
        line_end_pts = np.flipud(line_end_pts)
    
    # 2.1 compute intersections y-parallel axes
    x_min = np.min(line_end_pts[:, 0])
    x_max = np.max(line_end_pts[:, 0])
    dir_x = line_end_pts[1, 0]-line_end_pts[0, 0]
    
    x_intersects = []
    if (abs(dir_x) > tol): # False if line is amlost vertical
        x_axes = np.linspace(-R, R, npixels + 1)[1:-1]
        x_isections = x_axes[(x_axes > x_min)*(x_axes < x_max)]
        for x_section in x_isections:
            alpha = (x_section - line_end_pts[0, 0])/dir_x
            y_section = (1.0 - alpha)*line_end_pts[0, 1] + alpha*line_end_pts[1, 1]
            x_intersects.append([x_section, y_section, alpha]) 
    
    # 2.2 compute intersections y-axes
    y_min = np.min(line_end_pts[:, 1])
    y_max = np.max(line_end_pts[:, 1])
    dir_y = (line_end_pts[1, 1]-line_end_pts[0, 1])
    
    y_intersects = []
    if (abs(dir_y) > tol): # False if line is amlost horizontal
        y_axes = np.linspace(-R, R, npixels + 1)[1:-1]
        y_isections = y_axes[(y_axes > y_min)*(y_axes < y_max)]
        for y_section in y_isections:
            alpha = (y_section - line_end_pts[0, 1])/dir_y
            x_section = (1.0 - alpha)*line_end_pts[0, 0] + alpha*line_end_pts[1, 0]
            y_intersects.append([x_section, y_section, alpha])
    
    # 3. merge interesections with axes while preserving direction orderliness     
    
    # make both merging lists to monotonically increase in alpha 
    if (dir_x < 0.0):
        x_intersects.reverse()
    if (dir_y < 0.0):
        y_intersects.reverse()
    
    len_x_intersect = len(x_intersects)
    len_y_intersect = len(y_intersects)
    ind_x = 0
    ind_y = 0
    pts_intersect = [] # list of points where line intersects the grid
    
    # assembling pairs (x,y) on the line of integration
    # insert starting point
    pts_intersect.append([line_end_pts[0, 0], line_end_pts[0, 1],  0])
    #insert middle points
    while (ind_x < len_x_intersect and ind_y < len_y_intersect):            
        if (x_intersects[ind_x][-1] < y_intersects[ind_y][-1]):
            pts_intersect.append(x_intersects[ind_x])
            ind_x += 1
            continue
        if (x_intersects[ind_x][-1] > y_intersects[ind_y][-1]):
            pts_intersect.append(y_intersects[ind_y])
            ind_y += 1
            continue
        if (x_intersects[ind_x][-1] == y_intersects[ind_y][-1]):
            pts_intersect.append(x_intersects[ind_x])
            ind_x += 1
            ind_y += 1

    if (ind_x == len_x_intersect):
            assert ind_y < len_y_intersect, "Failed to attach the rest of ordered y-array"
            pts_intersect.extend(y_intersects[ind_y:])
    if (ind_y == len_y_intersect):
            assert ind_x < len_x_intersect, "Failed to attach the rest of ordered x-array"
            pts_intersect.extend(x_intersects[ind_x:])
    # insert finishing point
    pts_intersect.append([line_end_pts[1, 0], line_end_pts[1, 1],  1])

    # 4. compute the value of the line integral
    row_container = np.zeros((npixels, npixels))
    len_intersection = np.linalg.norm(line_end_pts[1]-line_end_pts[0])
    
    for ind in range(len(pts_intersect)-1):
        pix_len = (pts_intersect[ind + 1][-1] - pts_intersect[ind][-1])
        mid_x = 0.5*(pts_intersect[ind + 1][0] + pts_intersect[ind][0])
        mid_y = 0.5*(pts_intersect[ind + 1][1] + pts_intersect[ind][1])

        ind_x = int((mid_x + R) / dx) # get pixel indices
        ind_y = (npixels-1) - int((mid_y + R)/dx)
        
        if ((ind_x >= 0 and ind_x < npixels) and (ind_y >= 0 and ind_y < npixels)):
            row_container[ind_y, ind_x] = pix_len*len_intersection
    
    matrix[row_ind] = row_container.ravel()

@nb.njit(cache=True, parallel=True)
def get_projection_matrix(thetas, shifts, npixels, radius, int_tol=1e-15,
                        verbose=False,
                        warnings=False):
    nproj = len(thetas)
    nshifts = len(shifts)

    projector_matrix = np.zeros((len(thetas)*len(shifts), npixels**2))
    for itheta in range(nproj):
        for ishift in range(nshifts):
            row_ind = itheta*nshifts + ishift
            _fill_projector_row(
                projector_matrix,
                row_ind,
                thetas[itheta], shifts[ishift], radius, 
                int_tol, 
                verbose, 
                warnings)
            
    return projector_matrix

def get_and_save_projector(path, resolution):
    assert path[-3:] == 'npz'
    
    nproj = 2*resolution
    nshifts = resolution
    dom_radius = 1.0 # radius of the domain

    # prepare projector, backprojector, senisitivities
    ds = 2*dom_radius/resolution
    thetas = np.linspace(0, np.pi, nproj, endpoint=False)  # note : 0 and n.pi are the same 
    shifts = np.linspace(-1. + ds/2., 1. - ds/2., nshifts) # do not take lines which touch image border

    # compute Radon matrix and normalize it, so that it is stochastic column-wise
    projector = get_projection_matrix(thetas, shifts, resolution, dom_radius) # get Radon matrix
    _sensitivities = np.sum(projector, axis=0).reshape((resolution*resolution,))

    projector = np.divide(projector, _sensitivities)
    assert np.all(np.allclose(projector.sum(axis=0), np.ones(resolution*resolution))), "Failed to normalize the projector"
    
    np.savez(path, projector=projector)

    return

