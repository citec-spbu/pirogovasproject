import type { User } from '../../entities/user/model/types';
import eyeIcon from '../../shared/assets/icons/eyeIcon.svg';
import editIcon from '../../shared/assets/icons/editIcon.svg'
import deleteIcon from '../../shared/assets/icons/deleteIcon.svg'
import cls from './UsersTable.module.scss';

interface UsersTableProps {
  users: User[];
  onViewUser: (user: User) => void;
  onEditUser: (user: User) => void;
  onDeleteUser: (user: User) => void;
}

export const UsersTable = ({
  users,
  onViewUser,
  onEditUser,
  onDeleteUser,
}: UsersTableProps) => {
  if (users.length === 0) {
    return <p className={cls.empty}>Пользователи не найдены</p>;
  }

  return (
    <div className={cls.tableWrapper}>
      <table className={cls.table}>
        <thead>
          <tr>
            <th>ФИО</th>
            <th>Логин</th>
            <th>Организация</th>
            <th>Действия</th>
          </tr>
        </thead>

        <tbody>
          {users.map((user) => (
            <tr key={user.id}>
              <td>{user.fullName}</td>
              <td>{user.login}</td>
              <td>{user.organization}</td>
              <td>
                <div className={cls.actions}>
                  <button
                    type="button"
                    className={cls.iconButton}
                    onClick={() => onViewUser(user)}
                    aria-label="Посмотреть пользователя"
                  >
                    <img src={eyeIcon} alt="viewuser" className={cls.eyeIcon} />
                  </button>

                  <button
                    type="button"
                    className={cls.editButton}
                    onClick={() => onEditUser(user)}
                    aria-label="Редактировать пользователя"
                  >
                    <img src={editIcon} alt="edituser" className={cls.editIcon} />
                  </button>

                  <button
                    type="button"
                    className={cls.deleteButton}
                    onClick={() => onDeleteUser(user)}
                    aria-label="Удалить пользователя"
                  >
                    <img src={deleteIcon} alt="deleteuser" className={cls.deleteIcon} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};