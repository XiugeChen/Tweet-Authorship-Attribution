
RAW_FILE_FOLDER = "../resources/data/"
RAW_FILE_NAME = "train_tweets.txt"
RAW_FILE = RAW_FILE_FOLDER + RAW_FILE_NAME


def extract_users_from_source(number_of_users=100):
    """Extra first given number of users' twitter from the source file, and 
    put them into a new file.
    """
    new_file_name = "{0}{1}_{2}".format(
        RAW_FILE_FOLDER, number_of_users, RAW_FILE_NAME)

    with open(RAW_FILE, "r") as raw_file:
        file_rows = raw_file.readlines()
        users_extracted = []

        with open(new_file_name, "w") as target_file:
            for row in file_rows:
                user_id = row.split("\t")[0]
                if user_id not in users_extracted:
                    users_extracted.append(user_id)

                if (len(users_extracted) >= number_of_users):
                    return

                target_file.write(row)


extract_users_from_source(50)
extract_users_from_source(100)
extract_users_from_source(200)
extract_users_from_source(500)
extract_users_from_source(1000)
