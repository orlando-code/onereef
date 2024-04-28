def create_directory_structure(base_path, dirs):
    for dir in dirs:
        dir_path = base_path / dir

        # Check if the dir already exists
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")