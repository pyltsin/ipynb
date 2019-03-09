
def get_list(user_data,site_freq,\
             session_length=10, window_size=10):
#         from IPython.core.debugger import Tracer; Tracer()() 
        N=user_data.shape[0]
        r_b = lambda x: x+session_length
        slice_list=((i,r_b(i) if r_b(i)<N else N) for i in range(0,N,window_size))
        
        list_sites = []
        
        for pair in slice_list:
            lst = list(\
                map(lambda x: site_freq[x][0],user_data.site.values[pair[0]:pair[1]]))
            list_sites.append(lst)
        return list_sites
    
def prepare_sparse_train_set_window(path_to_csv_files, site_freq_path, 
                                    session_length=10, window_size=10):
    
    ''' ВАШ КОД ЗДЕСЬ'''
    pat = re.compile("user([\d]+)[.]")
    indptr = [0]
    count = 0
    indices = []
    data = []
    y = []

    for u_id,f in tqdm(enumerate(sorted(glob(path_to_csv_files+'/*')))):
        user_data = pd.read_csv(f)
        lst_ = get_list(user_data,site_freq_path,session_length,window_size)
        for col in lst_:
            counter = Counter(col)
            count += len(counter)
            indptr.append(count)
            indices += counter.keys()
            data += counter.values()
        y.append([int(re.search(pat,f).group(1))]*len(lst_))
    # X = csr_matrix((data, indices, indptr), shape=(X.shape[0], n))[:,1:]