import { useState } from 'react';
import starFullIcon from '../../shared/assets/icons/starFullIcon.svg';
import starEmptyIcon from '../../shared/assets/icons/starEmptyIcon.svg';
import cls from './ReportReview.module.scss';
import editIcon from '../../shared/assets/icons/editIcon.svg'

interface ReportReviewProps {
  disabled?: boolean;
  initialRating?: number | null;
  initialComment?: string | null;
  onSubmit: (rating: number, comment: string) => Promise<void> | void;
}

export const ReportReview = ({
  disabled = false,
  initialRating = 0,
  initialComment = '',
  onSubmit,
}: ReportReviewProps) => {
  const [rating, setRating] = useState(initialRating ?? 0);
  const [comment, setComment] = useState(initialComment ?? '');
  const [isCommentOpen, setIsCommentOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successText, setSuccessText] = useState('');
  const [errorText, setErrorText] = useState('');

  const handleRatingClick = (value: number) => {
    if (disabled) return;

    setRating(value);
    setIsCommentOpen(true);
    setSuccessText('');
    setErrorText('');
  };

  const handleSubmit = async () => {
    if (disabled || rating === 0) return;

    setIsSubmitting(true);
    setSuccessText('');
    setErrorText('');

    try {
      await onSubmit(rating, comment);

      setSuccessText('Отзыв сохранён');
      setIsCommentOpen(false);
    } catch (error) {
      console.error('Не удалось сохранить отзыв:', error);
      setErrorText('Не удалось сохранить отзыв');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={cls.review}>
      <div className={cls.topRow}>
        <div className={cls.stars}>
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              className={cls.starButton}
              disabled={disabled}
              onClick={() => handleRatingClick(star)}
              aria-label={`Оценить на ${star}`}
            >
              <img
                src={star <= rating ? starFullIcon : starEmptyIcon} alt="" className={cls.starIcon}/>
            </button>
          ))}
        </div>

        <button
          type="button"
          className={cls.commentButton}
          disabled={disabled}
          onClick={() => setIsCommentOpen((prev) => !prev)}
          aria-label="Написать отзыв"
        >
          <img src={editIcon} alt="" className={cls.editIcon} />
        </button>
      </div>

      {isCommentOpen && (
        <div className={cls.commentPanel}>
          <textarea
            className={cls.textarea}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="Напишите отзыв..."
          />

          <button
            type="button"
            className={cls.submitButton}
            onClick={handleSubmit}
            disabled={rating === 0 || isSubmitting}
          >
            {isSubmitting ? '...' : 'Отправить'}
          </button>
        </div>
      )}

      {successText && <p className={cls.successText}>{successText}</p>}
      {errorText && <p className={cls.errorText}>{errorText}</p>}
    </div>
  );
};