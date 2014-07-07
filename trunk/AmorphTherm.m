%     <A program for calculating amorphous material thermal transport properties using lattice dynamics>
%         This program aim to provide an efficient way to evaluate
%         thermal transport properties using lattice dynamics approaches.
%
%         Thermophysical parameters like diffusivity and thermal conductivity can by obtained by simply feeding
%         an equilbrium structure as the input.
%         Anharmonic interactions have also been considered in this
%         program, which is an important feature of it. This program also
%         allows the user to take advantage of the GPU computing to reduce
%         the time spent on eigenvalue problem solving step.
%
%
%     Copyright (C) 2014  
%                Version:    3.6
%                 Author:    Zhi Guo, PhD. 
%                  Email:    zguo1@nd.edu 
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 2 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

function [eigval_col, dfs, kapa]=AmorphTherm(input_coord, input_hessian, T, anharm_mode_no)

    mat_dim=size(input_hessian,1);
    num_of_particles=mat_dim/3;
    % First diagonalize the hessian matrix
    % eigvec is eigenvectors associated with the eigen values
    
    fprintf('INFO: Solving eigenvalue problem...');
    
    %{
    gpu_input_hessian=gpuArray(input_hessian);
    [gpu_eigvec, gpu_eigval]=eig(gpu_input_hessian);
    eigvec=gather(gpu_eigvec);
    eigval=gather(gpu_eigval);
    
    clear gpu_input_hessian;
    clear gpu_eigvec;
    clear gpu_eigval;
    
    disp('INFO: GPU operation done...');
    %}
    
    [eigvec, eigval]=eig(input_hessian);
    disp('done');

    % plot out the density of state

    eigvec=single(eigvec);
    eigval=single(eigval);
    input_coord=single(input_coord);
    input_hessian=single(input_hessian);
    
    eigval_col=diag(eigval);
    eigval_col=eigval_col';
    %{
    
    % The by-element assignment might be inefficient
    for i=1:mat_dim
        eigval_col(i)=eigval(i,i);
    end
    %}

    n=hist(eigval_col, 50);

    % construct distance matrix

    fprintf('INFO: Constructing distance matrix...');

    %comb_coord=reshape(input_coord', mat_dim, 1);
    input_coord=input_coord';
    
    distance_mat=zeros(mat_dim, mat_dim);
    
    
    a=zeros(3, num_of_particles, num_of_particles);
    
    for i=1:3
        a(i,:,:)=repmat(squeeze(input_coord(i,:)),num_of_particles,1);
        rot_a=squeeze(a(i,:,:));
        a(i,:,:)=rot_a-rot_a';
    end
    
    clear rot_a;
    a=a.*(-1);
    
    
    unit_cell1=[1 0 0; 0 0 0; 0 0 0];
    unit_cell2=[0 0 0; 0 1 0; 0 0 0];
    unit_cell3=[0 0 0; 0 0 0; 0 0 1];
    
    tmp=squeeze(a(1,:,:));       
    distance_mat=kron(tmp, unit_cell1);
    tmp=squeeze(a(2,:,:));
    distance_mat=distance_mat+kron(tmp,unit_cell2);
    tmp=squeeze(a(3,:,:));
    distance_mat=distance_mat+kron(tmp,unit_cell3);
    clear tmp;
    clear a;

    disp('done');


    % transforming real space deformation work into normal mode space
    fprintf('INFO: Calculating heat flux operator...');

    trans_eigvec=eigvec';
    %trans_eigvec=inv(eigvec);
    deform_work=input_hessian*distance_mat;

    clear distance_mat;
    %coupling_ij=trans_eigvec*deform_work*eigvec;
    coupling_ij=eigvec*deform_work*trans_eigvec;
    clear deform_work;
    clear trans_eigvec;

    % weighing elements by frequency dependent coefficient 

    a=eigval_col(7:mat_dim);
    denom_cofactor=sqrt(kron(a', a));
    clear a;
    
    for i=1:(mat_dim-6)
        sum_omega(:, i)=eigval_col(7:mat_dim)+eigval_col(i+6);
    end
    

    cofactor=sum_omega./denom_cofactor;
    clear sum_omega;
    clear denom_cofactor;
    heat_flux_mat=coupling_ij(7:mat_dim, 7:mat_dim).*cofactor;
    clear cofactor;
    clear coupling_ij;
    
    squ_heat_flux=heat_flux_mat.^2;
    clear heat_flux_mat;
    
    % This step is important!
    % The summation iterates over all non-diagonal elements!
    
    squ_heat_flux(logical(eye(mat_dim-6)))=0;
    
    %tolerance=(0.01)*(eigval_col(7));
    tolerance=(0.1)*(eigval_col(7));
    %tolerance=1;
    
    eigval_col=eigval_col(7:end);
    
    for i=1:mat_dim-6
        center_value=eigval_col(i);
        
        delta=(eigval_col-center_value)*2/tolerance;
        use_eigval=1./(1+delta.^2);
        clear delta;
        
        
        
        %squ_col=squ_heat_flux(:,i);
        squ_col=squ_heat_flux(i,:);
        squ_col=squ_col.*use_eigval;
        dfs(i)=sum(squ_col)/(eigval_col(i)^2);
    end
    clear squ_heat_flux;

    disp('done');

    fprintf('INFO: Calculating diffusivity ... ');
    
    for i=1:mat_dim-6
        %208.5 cm-1 at 300 K
        boltzman=exp(eigval_col(i)/208.5*(T/300));
        heat_capacity(i)=((eigval_col(i)/T)^2)*(boltzman/((boltzman-1)^2));
    end
    
    kapa=dfs*(heat_capacity')/(num_of_particles);
    filename=['eigval_', num2str(T) ,'.dat'];
    eigval_flip=eigval_col';
    dlmwrite(filename, eigval_flip); 
    dfs_flip=dfs';
    filename=['dfs_', num2str(T), '.txt'];
    dlmwrite(filename, [eigval_flip, dfs_flip]);
    clear eigval_flip;
    clear dfs_flip;

    disp('done');

    fprintf('INFO: Calculating participation ratio ... ');
    
    % Added on Apr 24th, calculate the participation ratio
    % First calculate the coefficient squares and sum them

    sqr_eigvec=eigvec.*eigvec;
    sum_amp=sum(sqr_eigvec.*sqr_eigvec);
    sum_amp=mat_dim*sum_amp;
    sum_amp=1./sum_amp;    
    sum_amp=sum_amp(7:end);
    filename=['par_ratio_', num2str(T), '.txt'];
    dlmwrite(filename, sum_amp');
    clear sum_amp;
    disp('done');
    
    % Calculate the Localization factor and norm of direction vector
    fprintf('INFO: Calculating correlation of direction vectors (by mode) ... ');
    
    % First calculate the direction vector
    
    A_vec=get_dir_vec_norm(eigvec(:,7:end));
    Cid=calc_corr(single(A_vec(:,:,1:anharm_mode_no)), 100, 50);
    filename=['Cid_', num2str(T), '.txt'];
    dlmwrite(filename, Cid);
    clear Cid;
    disp('done');
    
    % Calculate the anharmonic decay rates, W
    fprintf('INFO: Calculating anharmonic decay rate ... ');
    
    grin_const=1.0;
    
    W=anharm(grin_const,anharm_mode_no*3);
    filename=['anharm_',num2str(T), '.txt'];
    dlmwrite(filename, W);
    disp('done');

    % Calculate localization vector

    fprintf('INFO: Calculating localization vector ... ');
    
    local_len=calc_local(eigvec);
    local_len=local_len(7:end);
    filename=['local_len_', num2str(T), '.txt'];
    dlmwrite(filename, local_len);

    disp('done');
    

    % Calculate cross over time from localization vectors 

    fprintf('INFO: Calculating cross over time from localization vectors ... ');

    border_time=(1/3)*((local_len.^2)./dfs);
    filename=['crosstime_', num2str(T), '.txt'];
    dlmwrite(filename, border_time);
    
    disp('done');
    
    % Comparing mode decay life time and the cross over time
    
    mode_decay_t=(1.0)./(W);
    min_set=size(mode_decay_t);
    %plot(eigval_col(1:min_set), mode_decay_t, eigval_col(1:min_set), border_time(1:min_set));
    
    % definition of the function of obtaining the diretion vector norm and
    % the displacement vector
    function A_vec=get_dir_vec_norm(input_eigvec)
        % we assume that the input matrix has already exclude the first 6
        % column vectors representing translation and rotation

        
        x_comp=input_eigvec(1:3:end,:);
        y_comp=input_eigvec(2:3:end,:);
        z_comp=input_eigvec(3:3:end,:);
        
        dir_vec_norm=sqrt(x_comp.^2+y_comp.^2+z_comp.^2);
        
        A_vec(1,:,:)=x_comp./dir_vec_norm;
        A_vec(2,:,:)=y_comp./dir_vec_norm;
        A_vec(3,:,:)=z_comp./dir_vec_norm;
        
        
    end

    function Cid=calc_corr(A_vec, dmax, num_of_points)
        
        atom_num=size(A_vec,2);
        mode_num=size(A_vec,3);
       
        % construction logical matrix based on the distance between atom A
        % and atom B
        
        cut_err=(dmax/num_of_points)/2;

        src=repmat(single(input_coord), [1 1 atom_num]);
        dist_ab=single(zeros(atom_num, atom_num));
        for s=1:3
            src_sub=squeeze(src(s,:,:));
            delta_ab=src_sub-src_sub';
            dist_ab=dist_ab+delta_ab.^2;
        end
        clear delta_ab;
        clear src;
        clear src_sub;
        dist_ab=sqrt(dist_ab);
        
        d_arr_cmp=linspace(0, dmax, num_of_points);
        d_arr_cmp=single(d_arr_cmp);

        test_set=repmat(d_arr_cmp', [1 atom_num atom_num])-permute(repmat(dist_ab,[1 1 num_of_points]), [3 1 2]);    % dubious code
        clear d_arr_cmp;
        clear dist_ab;
        weight=exp(-(test_set/cut_err).^2);
        clear test_set;
        
        vec_dot=single(zeros(atom_num, atom_num, mode_num));
        for s=1:mode_num
            sub_A=A_vec(:,:,s);
            vec_dot(:,:,s)=triu((sub_A')*sub_A);
        end
        clear sub_A;
        sum_of_weight=sum(sum(weight,3),2);        
        sum_of_corr=single(zeros(num_of_points,mode_num));
        sum_of_corr=sum_of_corr+reshape(weight,num_of_points,[])*reshape(vec_dot,atom_num*atom_num,[]);
        clear vec_dot;
        clear weight;
        
	%{
        for a_idx=1:atom_num
            for b_idx=(a_idx+1):atom_num
                    delta_ab=squeeze(input_coord(:,a_idx)-input_coord(:,b_idx));
                    dist_ab=sqrt(sum(delta_ab.^2));
                    test_set=d_arr_cmp-dist_ab;
                    satisfy_idx=find(abs(test_set)<cut_err,1);
                    if(~isempty(satisfy_idx))
                        vec_dot=squeeze(dot(A_vec(:,a_idx,:),A_vec(:,b_idx,:)));
                        weight=exp(-(test_set(satisfy_idx)/cut_err)^2);
                        sum_of_weight=sum_of_weight+weight;
                        sum_of_corr(satisfy_idx,:)=sum_of_corr(satisfy_idx,:)+(vec_dot')*2*weight;
                    end
            end
        end
	%}
        
        Cid=sum_of_corr./repmat(sum_of_weight,[1 mode_num]);  
    end

    function rate=anharm(grin_const, total_len)
        counter=0;
        counter_last=1;
        counter_real=0;
        if total_len>size(eigval_col)
            total_len=size(eigval_col);
        end
        tol=0.01*eigval_col(1);
        for s=total_len:-1:1
            for p=s-1:-1:1
                target=eigval_col(s)-eigval_col(p);
                if target>eigval_col(p)
                    break;
                end
                for k=1:total_len
                    if (abs(target-eigval_col(k))<tol) 
                        counter=counter+1;
                        location(counter,:)=[s p  k eigval_col(s) eigval_col(p) eigval_col(k)];
                        break;
                    end
                    if target-eigval_col(k)<0
                        break;
                    end
                end
            end
            tmp=location(counter_last:counter,4:6);
            counter_last=counter;
            all_item=size(tmp,1);

	    if all_item~=0

            %{
            tmp2=tmp(:,2).*tmp(:,3);
            tmp3=((1+1/(exp(tmp(:,2)/210)-1)+1/(exp(tmp(:,3)/210)-1))')./tmp2;
            counter_real=counter_real+1;
            rate(counter_real)=sum(tmp3);
            %}
            accum=0;
            for m=1:all_item
                kbTT=208.5*T/300;
                accum=accum+(1+1/(exp(tmp(m,2)/kbTT)-1)+1/(exp(tmp(m,3)/kbTT)-1))/(tmp(m,2)*tmp(m,3));
            end
            counter_real=counter_real+1;
	    rate(1,counter_real)=eigval_col(s);
            rate(2,counter_real)=accum/eigval_col(s)*(grin_const^2);
	    counter_last=counter_last+1; 
	
	    end
        end
    end

    function local_len=calc_local(input_eigvec)
        % we assume that the input matrix has already exclude the first 6
        % column vectors representing translation and rotation

	kbt=208.5*T/300;
        coeff=sqrt(kbt)./(2*pi*diag(eigval));
        coeff=repmat(coeff, 1, num_of_particles);
        coeff=coeff';
        
        x_comp=input_eigvec(1:3:end,:).*coeff;
        y_comp=input_eigvec(2:3:end,:).*coeff;
        z_comp=input_eigvec(3:3:end,:).*coeff;
        
        dir_vec_norm=sqrt(x_comp.^2+y_comp.^2+z_comp.^2);
        tmp_a=input_coord*dir_vec_norm;
        tmp_b=sum(dir_vec_norm,1);
        
        local_center(1,:)=tmp_a(1,:)./tmp_b;
        local_center(2,:)=tmp_a(2,:)./tmp_b;
        local_center(3,:)=tmp_a(3,:)./tmp_b;
        

        delta_dist=repmat(input_coord, [1 1 mat_dim])-permute(repmat(local_center, [1 1 num_of_particles]), [1 3 2]);
        delta_dist=delta_dist.^2;
        delta_dist=squeeze(sum(delta_dist,1));
        delta_dist=delta_dist.*dir_vec_norm;
        delta_dist=squeeze(sum(delta_dist,1));
        
        local_len=delta_dist./tmp_b;
        local_len=sqrt(local_len);
        
    end

end

